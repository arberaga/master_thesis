"""Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Reference Code: https://github.com/tkipf/relational-gcn
"""
import argparse
import time
import optuna
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.data.rdf import AIFBDataset, AMDataset, BGSDataset, MUTAGDataset
from model import EntityClassify
from sklearn.metrics import mean_absolute_percentage_error
from utils import EarlyStopping

def main2(trial, train_idx, val_idx):
    # load graph data
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
    
    
    else:
        raise ValueError()

    g = dataset[0]
    #print(g.nodes['house'].data['feat'])
    category = 'house'#dataset.predict_category
    #train_mask = g.nodes[category].data.pop("train_mask")
    test_mask = g.nodes[category].data.pop("test_mask")
    #train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
    test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()
    labels = g.nodes[category].data.pop("Price")


    print('In Dataset: {}, node types num: {}'.format(args.dataset, len(g.ntypes)))
    print('In Dataset: {}, edge types num: {}'.format(args.dataset, len(g.etypes)))

    for ntype in g.ntypes:
        print('There are total {} nodes in type {}'.format(g.number_of_nodes(ntype), ntype))

    for src, etype, dst in g.canonical_etypes:
        print('There are total {} edges in type {}'.format(g.number_of_edges((src, etype, dst)), (src, etype, dst)))
    # split dataset into train, validate, test
    # if args.validation:
    #     val_idx = train_idx[: len(train_idx) // 5]
    #     train_idx = train_idx[len(train_idx) // 5 :]
    # else:
    # val_idx = train_idx

    # check cuda
    use_cuda = args.gpu >= 0 and th.cuda.is_available()
    if use_cuda:
        th.cuda.set_device(args.gpu)
        g = g.to("cuda:%d" % args.gpu)
        labels = labels.cuda()
        # train_idx = train_idx.cuda()
        # test_idx = test_idx.cuda()

    hid_dims = trial.suggest_int("hid_dims", 4, 64)
    num_layers = trial.suggest_int("num_layers", 3, 5)
    num_basis_factor = trial.suggest_int("num_basis_factor", 1, 10)
    drop_outs = trial.suggest_float("drop_outs", 0, 0.3)
    lrs = trial.suggest_float("lrs", 0.001, 0.01)

    # create model
    model = EntityClassify(
        g,
        hid_dims,
        1, #nr of outputs per node
        num_bases=num_basis_factor,
        num_hidden_layers=num_layers - 2,
        dropout=drop_outs,
        use_self_loop=args.use_self_loop,
    )

    if use_cuda:
        model.cuda()

    # optimizer
    optimizer = th.optim.Adam(
        model.parameters(), lr=lrs, weight_decay=args.l2norm
    )
    earlystoper = EarlyStopping(patience=10)

    # training loop
    print("start training...")
    dur = []
    model.train()
    for epoch in range(args.n_epochs):
        #print(category)
        optimizer.zero_grad()
        if epoch > 5:
            t0 = time.time()
        logits = model()[category]   # h={'house':g.nodes['house'].data['feat']})
        
        loss = F.mse_loss(logits[train_idx].reshape(-1, 1),labels[train_idx].reshape(-1, 1)) # changed from F.cross_entropy
        val_loss = F.mse_loss(logits[val_idx].reshape(-1, 1), labels[val_idx].reshape(-1, 1)) # changed from F.cross_entropy

        #print(loss)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t1 = time.time()

        if epoch > 5:
            dur.append(t1 - t0)
        # train_acc = th.sum(
        #     logits[train_idx] == labels[train_idx]
        # ).item() / len(train_idx)
        # val_acc = th.sum(
        #     logits[val_idx] == labels[val_idx]
        # ).item() / len(val_idx)

        if earlystoper.step(val_loss, model): 
            print("Early stopping at epoch {}".format(epoch))
            break
        
        print(
            "Epoch {:05d} | Train Loss: {:.4f} | Valid loss: {:.4f} | Time: {:.4f}".format(
                epoch,
                loss.item(),
                val_loss.item(),
                np.average(dur),
            )
        )

    model.load_state_dict(th.load('model.param'))

    model.eval()
    logits = th.flatten(model.forward()[category])
    mape_loss = mean_absolute_percentage_error(logits[val_idx].squeeze().cpu().detach().numpy(),labels[val_idx].cpu().detach().numpy())
    
    return mape_loss


def objective_cv(trial):
    from sklearn.model_selection import TimeSeriesSplit 
    ts_split = TimeSeriesSplit(n_splits=4)
    # Get the dataset.
    dataset = dgl.load_graphs("/home/arberaga/Desktop/MasterThesis/gcn_for_housing/taiwan_dataset/v1-CV/taiwan after minmaxing/taiwan after minmaxing.bin")[0][0]
    train_mask = dataset.nodes['house'].data.pop('train_mask')

    percentages = []
    for fold_idx, (train_idx, valid_idx) in enumerate(ts_split.split(range(len(train_mask)))):
        print("Fold ID:", fold_idx)
        mape = main2(trial, train_idx, valid_idx)
        percentages.append(mape)
        print(percentages)    
    return np.mean(percentages)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RGCN")
    parser.add_argument(
        "--dropout", type=float, default=0, help="dropout probability"
    )
    # parser.add_argument("--n-hidden", type=int, default=13, help="number of hidden units" )
    # parser.add_argument("--n-hidden", type=int, default=12, help="number of hidden units" )

    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    # parser.add_argument("--lr", type=float, default=8e-3, help="learning rate")
    # parser.add_argument(
    #     "--n-bases",
    #     type=int,
    #     default=2,
    #     help="number of filter weight matrices, default: -1 [use all]",
    # )
    # parser.add_argument(
    #     "--n-layers", type=int, default=4, help="number of propagation rounds"
    # )
    parser.add_argument(
        "-e",
        "--n-epochs",
        type=int,
        default=200,
        help="number of training epochs",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, default="own_w_parks", help="dataset to use"
    )
    parser.add_argument(
        "--model_path", type=str, default=None, help="path for save the model"
    )
    parser.add_argument("--l2norm", type=float, default=0, help="l2 norm coef")
    parser.add_argument(
        "--use-self-loop",
        default=True,
        action="store_true",
        help="include self feature as a special relation",
    )
    parser.add_argument("--testing", dest="validation", action="store_false")
    args = parser.parse_args()

    np.random.seed(123456)
    th.manual_seed(123456)

    # Run parameter tunning
    results = []
    from optuna.samplers import TPESampler

    # Add stream handler of stdout to show the messages
    # optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "rgcn-study"  # Unique identifier of the study.
    storage_name = "sqlite:///CV{}.db".format(study_name)

    study = optuna.create_study(study_name="CV"+study_name, storage=storage_name,sampler=TPESampler(), pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource=200, reduction_factor=3), load_if_exists=True, directions=["minimize","minimize"])
    study.optimize(objective_cv, n_trials=150, gc_after_trial=True)


