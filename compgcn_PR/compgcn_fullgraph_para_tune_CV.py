#-*- coding:utf-8 -*-

# Author:james Zhang
# Datetime:20-09-27 10:48
# Project: DGL
"""
    This file is designed for hyper-parameter tuning to achieve the best accuary performance in 4 DGL dataset:
    - AIDB, MUTAG, BGS, and AM

    1. Basically use early stop to get the best val-performance's hyper-parameters, and
    2. The final performance will use all training data and the hyper-parameters to get the test performance.

    Main hyper-parameters would invovle:
    0. The composition function: SUB, MUL, and CCORR
    1. Vector Basis for edge type;
    2. The number of layers of the CompGCN model;
    3. The number of hidden dimensionality;
    4. The dropout rate

"""
import optuna
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np
import pandas as pd
import argparse
import os
import dgl
import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from data_utils import build_dummy_comp_data
from models.model import CompGCN
from model_utils import EarlyStopping

def main2(trial, train_idx, val_idx):
    args = { "gpu": -1, "dataset" : "taiwan_after_minmax_cv" }
    # Step 1： Prepare graph data and split into train/validation ============================= #
    #dataset = dgl.data.CSVDataset('/home/arberaga/Desktop/Master Thesis/melbourne_dataset')
    if args["dataset"] == "melb_before_minmax":
        dataset = dgl.load_graphs("/home/arberaga/Desktop/MasterThesis/gcn_for_housing/melbourne_dataset/Without externals before minmax/melbourne before minmax/melbourne before minmax.bin")[0]
    elif args["dataset"] == "melb_after_minmax":
        dataset = dgl.load_graphs("/home/arberaga/Desktop/MasterThesis/gcn_for_housing/melbourne_dataset/Without externals after minmax/melbourne after minmax/melbourne after minmax.bin")[0]
    elif args["dataset"] == "melb_w_parks":
        dataset = dgl.load_graphs("/home/arberaga/Desktop/MasterThesis/gcn_for_housing/melbourne_dataset/with externals/melbourne after minmax with externals/melbourne after minmax with externals.bin")[0]
    elif args["dataset"] == "taiwan_after_minmax":
           dataset = dgl.load_graphs("/home/arberaga/Desktop/MasterThesis/gcn_for_housing/taiwan_dataset/v1/taiwan after minmaxing/taiwan after minmaxing.bin")[0]
    elif args["dataset"] == "taiwan_after_parks":
           dataset = dgl.load_graphs("/home/arberaga/Desktop/MasterThesis/gcn_for_housing/taiwan_dataset/v4/taiwan different/taiwan different.bin")[0]
    elif args["dataset"] == "taiwan_before_minmax":
        dataset = dgl.load_graphs("/home/arberaga/Desktop/MasterThesis/gcn_for_housing/taiwan_dataset/v0/taiwan before minmaxing/taiwan before minmaxing.bin")[0]
    
    # Load from hetero-graph
    heterograph = dataset[0]

    # number of classes to predict, and the node type
    num_classes = 1
    target = 'house'

    # basic information of the dataset
    num_rels = len(heterograph.canonical_etypes)
    # print(heterograph.nodes[target].data)

    #train_mask = heterograph.nodes[target].data.pop('train_mask')
    test_mask = heterograph.nodes[target].data.pop('test_mask')
    # val_mask = heterograph.nodes[target].data.pop('validate_mask')
    # print(heterograph.nodes[target].data)
    # print("validation masks aboev")

    # train_idx = th.nonzero(train_mask).squeeze()
    test_idx = th.nonzero(test_mask).squeeze()
    # val_idx = th.nonzero(val_mask).squeeze()
    labels = heterograph.nodes[target].data.pop('Price')
    

    # print('In Dataset: {}, node types num: {}'.format(args["dataset"], len(heterograph.ntypes)))
    # print('In Dataset: {}, edge types num: {}'.format(args["dataset"], len(heterograph.etypes)))

    # for ntype in heterograph.ntypes:
        # print('There are total {} nodes in type {}'.format(heterograph.number_of_nodes(ntype), ntype))

    # for src, etype, dst in heterograph.canonical_etypes:
        # print('There are total {} edges in type {}'.format(heterograph.number_of_edges((src, etype, dst)), (src, etype, dst)))

    # check cuda
    use_cuda = (args["gpu"] >= 0 and th.cuda.is_available())
    # print("If use GPU: {}".format(use_cuda))

    in_feat_dict = {}
    n_feats = {}

    # For node featureless, use an additional embedding layer to transform the data
    for ntype in heterograph.ntypes:
        n_feats[ntype] = th.arange(heterograph.number_of_nodes(ntype)) # empty tensor for nodes length
        in_feat_dict[ntype] = num_rels
        # added lines
        if(ntype == 'house' and "feat" in heterograph.nodes["house"].data.keys()):
            in_feat_dict[ntype] = len(heterograph.nodes["house"].data["feat"][0])
        if(ntype == 'park'):
            in_feat_dict[ntype] = len(heterograph.nodes["park"].data["feat"][0])
        if(ntype == 'hospital'):
            in_feat_dict[ntype] = len(heterograph.nodes["hospital"].data["feat"][0])
        if use_cuda:
            n_feats[ntype] = n_feats[ntype].to('cuda:{}'.format(args["gpu"]))

    if use_cuda:
        labels = labels.to('cuda:{}'.format(args["gpu"]))

    # Step 2: Create model =================================================================== #
    input_embs = nn.ModuleDict()
    for ntype in heterograph.ntypes:
        input_embs[ntype] = nn.Embedding(heterograph.number_of_nodes(ntype), num_rels)

    hid_dims = trial.suggest_int("hid_dims", 4, 64)
    num_layers = trial.suggest_int("num_layers", 3, 5)
    num_basis_factor = trial.suggest_int("num_basis_factor", 1, 10)
    comp_fns = trial.suggest_categorical("comp_fns",['sub', 'mul', 'ccorr'])
    drop_outs = trial.suggest_float("drop_outs", 0, 0.3)
    lrs = trial.suggest_float("lrs", 0.001, 0.01)
    compgcn_model = CompGCN(in_feat_dict=in_feat_dict,
                            hid_dim=hid_dims,
                            num_layers=num_layers,
                            out_feat=num_classes,
                            num_basis=num_basis_factor,
                            num_rel=num_rels,
                            comp_fn=comp_fns,
                            dropout=drop_outs,
                            activation=F.relu,
                            batchnorm=True
                            )

    if use_cuda:
        input_embs.to('cuda:{}'.format(args["gpu"]))
        compgcn_model = compgcn_model.to('cuda:{}'.format(args["gpu"]))
        heterograph = heterograph.to('cuda:{}'.format(args["gpu"]))

    # Step 3: Create training components ===================================================== #
    loss_fn = th.nn.MSELoss() #CrossEntropyLoss()
    optimizer = optim.Adam([
                            {'params': input_embs.parameters(), 'lr':lrs, 'weight_decay':5e-4},
                            {'params': compgcn_model.parameters(), 'lr':lrs, 'weight_decay':5e-4}
                            ])

    earlystoper = EarlyStopping(patience=10)
    
    # Step 4: training epoches =============================================================== #
    for epoch in range(200):

        
        # forward
        input_embs.train()
        compgcn_model.train()
        in_n_feats ={}
        for ntype, feat in n_feats.items():
            in_n_feats[ntype] = input_embs[ntype](feat)
        # added line
        if("house" in in_n_feats.keys() and "feat" in heterograph.nodes["house"].data.keys()):
            in_n_feats["house"] = heterograph.nodes["house"].data['feat']
        if("park" in in_n_feats.keys() and "feat" in heterograph.nodes["park"].data.keys()):
            in_n_feats["park"] = heterograph.nodes["park"].data['feat']
        if("hospital" in in_n_feats.keys() and "feat" in heterograph.nodes["hospital"].data.keys()):
            in_n_feats["hospital"] = heterograph.nodes["hospital"].data['feat']

        #print(in_n_feats.keys())
        logits = compgcn_model.forward(heterograph, in_n_feats)
        # print(epoch)
        # print(logits[target][:20])
        # print(labels[:20])
        # compute loss
        tr_loss = loss_fn(logits[target][train_idx].squeeze(), labels[train_idx])
        val_loss = loss_fn(logits[target][val_idx].squeeze(), labels[val_idx])

        # backward
        optimizer.zero_grad()
        tr_loss.backward()
        optimizer.step()

        #train_acc = th.sum(logits[target][train_idx].argmax(dim=1) == labels[train_idx]).item() / len(train_idx)
        #val_acc = th.sum(logits[target][val_idx].argmax(dim=1) == labels[val_idx]).item() / len(val_idx)
        print("Epoch {} | Train Loss: {:.4f} | Validation loss: {:.4f}".
              format( epoch, tr_loss.item(),  val_loss.item()))
        if earlystoper.step(val_loss, compgcn_model):
            print("Early stopping at epoch {}".format(epoch))
            break
    
    compgcn_model.load_state_dict(th.load("model.param"))
    input_embs.eval()
    compgcn_model.eval()

    in_n_feats = {}
    for ntype, feat in n_feats.items():
        in_n_feats[ntype] = input_embs[ntype](feat)
    # added line
    if("house" in in_n_feats.keys() and "feat" in heterograph.nodes["house"].data.keys()):
        in_n_feats["house"] = heterograph.nodes["house"].data['feat']
    if("park" in in_n_feats.keys() and "feat" in heterograph.nodes["park"].data.keys()):
        in_n_feats["park"] = heterograph.nodes["park"].data['feat']
    if("hospital" in in_n_feats.keys() and "feat" in heterograph.nodes["hospital"].data.keys()):
        in_n_feats["hospital"] = heterograph.nodes["hospital"].data['feat']

    logits = compgcn_model.forward(heterograph, in_n_feats)

    mape_loss = mean_absolute_percentage_error(logits[target][val_idx].squeeze().cpu().detach().numpy(),labels[val_idx].cpu().detach().numpy())
    print("MAPE: ",mape_loss)
    
    del compgcn_model
    del logits
    
    return mape_loss


def objective_cv(trial):
    from sklearn.model_selection import TimeSeriesSplit 
    ts_split = TimeSeriesSplit(n_splits=4)
    # Get the dataset.
    # select dataset as in if clauses above
    #dataset = dgl.load_graphs("/home/arberaga/Desktop/MasterThesis/gcn_for_housing/taiwan_dataset/v1-CV/taiwan after minmaxing/taiwan after minmaxing.bin")[0][0]
    train_mask = dataset.nodes['house'].data.pop('train_mask')

    percentages = []
    for fold_idx, (train_idx, valid_idx) in enumerate(ts_split.split(range(len(train_mask)))):
        print("Fold ID:", fold_idx)
        mape = main2(trial, train_idx, valid_idx)
        percentages.append(mape)
        print(percentages)    
    return np.mean(percentages)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BoSH CompGCN Full Graph')
    parser.add_argument("-d", "--dataset", type=str, required=True, help="dataset to use")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU Index")
    # disable below arguments and
    # parser.add_argument("--hid_dim", type=int, default=32, help="Hidden layer dimensionalities")
    # parser.add_argument("--num_layers", type=int, default=4, help="Number of layers")
    # parser.add_argument("--num_basis", type=int, default=40, help="Number of basis")
    parser.add_argument("--rev_indicator", type=str, default='_inv', help="Indicator of reversed edge")
    # parser.add_argument("--comp_fn", type=str, default='sub', help="Composition function")
    # parser.add_argument("--max_epoch", type=int, default=200, help="The max number of epoches")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    # parser.add_argument("--drop_out", type=float, default=0.1, help="Drop out rate")
    fp = parser.add_mutually_exclusive_group(required=False)
    fp.add_argument('--validation', dest='validation', action='store_true')
    fp.add_argument('--testing', dest='validation', action='store_false')
    parser.set_defaults(validation=True)

    args = parser.parse_args()
    # print(args)

    np.random.seed(123456)
    th.manual_seed(123456)

    # HP tunning ranges
    
    # Run parameter tunning
    from optuna.samplers import TPESampler

    # Add stream handler of stdout to show the messages
    # optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "compgcn-study"  # Unique identifier of the study.
    storage_name = "sqlite:///CV{}.db".format(study_name)

    study = optuna.create_study(study_name="CV"+study_name, storage=storage_name,sampler=TPESampler(), pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource=200, reduction_factor=3), load_if_exists=True, directions=["minimize","minimize"])
    study.optimize(objective_cv, n_trials=150, gc_after_trial=True)
