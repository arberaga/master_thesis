

# Import necessary libraries for calculating mean squared error and using the LightGBM regressor.
from sklearn.metrics import mean_squared_error as mse
from lightgbm import LGBMRegressor
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_absolute_percentage_error
import optuna

# Create an instance of the LightGBM Regressor with the RMSE metric.
ordinal_enc = OrdinalEncoder()
def main2(trial, train, train_id, val_id):
    max_depths = trial.suggest_int("max_depth", 5, 25)
    num_leavess = trial.suggest_int("num_leaves", 20, 50)
    learning_rates = trial.suggest_float("learning_rate", 0.001, 0.1)
    colsample_bytrees = trial.suggest_float("colsample_bytree", 0.7, 1)

    model = LGBMRegressor(metric='mse',num_iterations=200,max_depth=max_depths,num_leaves=num_leavess,learning_rate=learning_rates,colsample_bytree=colsample_bytrees)
    
    y_train = train.loc[train_id]['Price']
    X_train = train.loc[train_id].drop(["Price"], axis=1)
    y_val = train.loc[val_id]['Price']
    X_val = train.loc[val_id].drop(["Price"], axis=1)
    import re
    X_train = X_train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    X_val = X_val.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    # print(X_train)
    # Train the model using the training data.
    model.fit(X_train, y_train)
    
    # Make predictions on the training and validation data.
    logists_train = model.predict(X_train)
    logists_val = model.predict(X_val)

    # Calculate and print the Root Mean Squared Error (RMSE) for training and validation predictions.
    print("Training MAPE: ", mean_absolute_percentage_error(logists_train, y_train))
    print("Val MAPE: ", mean_absolute_percentage_error(logists_val, y_val))
    return mean_absolute_percentage_error(logists_val, y_val)

def objective_cv(trial):
    from sklearn.model_selection import TimeSeriesSplit 
    ts_split = TimeSeriesSplit(n_splits=4)
    # Get the dataset.
    dataset = pd.read_csv('/home/arberaga/Desktop/MasterThesis/gcn_for_housing/melbourne-csv-creation/melb w duplicates w parks and hospitals.csv')
    #dataset['Encoded_city'] = ordinal_enc.fit_transform(dataset[['City']]).astype(int)
    # dataset['Encoded_purpose'] = ordinal_enc.fit_transform(dataset[['The main purpose']]).astype(int)
    # dataset['Encoded_floor'] = ordinal_enc.fit_transform(dataset[['Floor type/floor height']]).astype(int)
    
    dummies_1 = pd.get_dummies(dataset.Suburb, prefix="Suburb")
    dataset = pd.concat([dataset, dummies_1], axis=1)
    print(dummies_1)
    dummies_1 = pd.get_dummies(dataset.Type, prefix="Type")
    dataset = pd.concat([dataset, dummies_1], axis=1)
    print(dummies_1)
    dummies_1 = pd.get_dummies(dataset.Method, prefix="Method")
    dataset = pd.concat([dataset, dummies_1], axis=1)
    print(dummies_1)
    dummies_1 = pd.get_dummies(dataset.SellerG, prefix="SellerG")
    dataset = pd.concat([dataset, dummies_1], axis=1)
    print(dummies_1)
    dummies_1 = pd.get_dummies(dataset.CouncilArea, prefix="CouncilArea")
    dataset = pd.concat([dataset, dummies_1], axis=1)
    print(dummies_1)
    dummies_1 = pd.get_dummies(dataset.Regionname, prefix="Regionname")
    dataset = pd.concat([dataset, dummies_1], axis=1)
    print(dummies_1)
    print(dataset.columns)
    dataset.drop(["Suburb", "Address","Type","Method","SellerG","Date","CouncilArea","Regionname"],axis=1,inplace=True)


    dataset.drop(["parks_nearby_0_place_id"],axis=1,inplace=True)
    dataset.drop(["parks_nearby_1_place_id"],axis=1,inplace=True)
    dataset.drop(["parks_nearby_2_place_id"],axis=1,inplace=True)
    dataset.drop(["parks_nearby_3_place_id"],axis=1,inplace=True)
    dataset.drop(["parks_nearby_4_place_id"],axis=1,inplace=True)
    dataset.drop(["hospital_nearby_0_place_id"],axis=1,inplace=True)
    dataset.drop(["hospital_nearby_1_place_id"],axis=1,inplace=True)
    dataset.drop(["hospital_nearby_2_place_id"],axis=1,inplace=True)
    dataset.drop(["hospital_nearby_3_place_id"],axis=1,inplace=True)
    dataset.drop(["hospital_nearby_4_place_id"],axis=1,inplace=True)
    train = dataset[:int(len(dataset)*0.875)]
    test = dataset[int(len(dataset)*0.875):]    
    #train_mask = dataset.nodes['house'].data.pop('train_mask')

    percentages = []
    for fold_idx, (train_idx, valid_idx) in enumerate(ts_split.split(range(len(train)))):
        # print("Fold ID:", train_idx)
        # print("Fold ID:", valid_idx)
        mape = main2(trial, train, train_idx, valid_idx)
        percentages.append(mape)
        print(percentages)    
    return np.mean(percentages)

if __name__ == '__main__':
    # Run parameter tunning
    from optuna.samplers import TPESampler

    # Add stream handler of stdout to show the messages
    study_name = "-lgbm_melbourne-study"  # Unique identifier of the study.
    storage_name = "sqlite:///CV{}.db".format(study_name)

    study = optuna.create_study(study_name="CV"+study_name, storage=storage_name,sampler=TPESampler(), pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource=200, reduction_factor=3), load_if_exists=True, direction="minimize")
    study.optimize(objective_cv, n_trials=150, gc_after_trial=True)