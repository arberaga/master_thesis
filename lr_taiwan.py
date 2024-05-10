

# Import necessary libraries for calculating mean squared error and using the LightGBM regressor.
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_absolute_percentage_error

# Create an instance of the LightGBM Regressor with the RMSE metric.
ordinal_enc = OrdinalEncoder()
def main2(train, train_id, val_id):

    model = LinearRegression()
    
    y_train = train.loc[train_id]['Total price (10,000 yuan)']
    X_train = train.loc[train_id].drop(["Total price (10,000 yuan)"], axis=1)
    y_val = train.loc[val_id]['Total price (10,000 yuan)']
    X_val = train.loc[val_id].drop(["Total price (10,000 yuan)"], axis=1)
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

def objective():
    from sklearn.model_selection import TimeSeriesSplit 
    ts_split = TimeSeriesSplit(n_splits=4)
    # Get the dataset.
    dataset = pd.read_csv('/home/arberaga/Desktop/MasterThesis/gcn_for_housing/taiwan-csv-creation/taiwan_with_location_with_both.csv')

    # dummies_1 = pd.get_dummies(dataset.Suburb, prefix="Suburb")
    # dataset = pd.concat([dataset, dummies_1], axis=1)
    # dummies_1 = pd.get_dummies(dataset.Type, prefix="Type")
    # dataset = pd.concat([dataset, dummies_1], axis=1)
    # dummies_1 = pd.get_dummies(dataset.Method, prefix="Method")
    # dataset = pd.concat([dataset, dummies_1], axis=1)
    dummies_1 = pd.get_dummies(dataset.City, prefix="City")
    dataset = pd.concat([dataset, dummies_1], axis=1)
    dataset['Proportion of main building'] = dataset['Proportion of main building'].str.rstrip('%').astype('float')

    dataset.drop(["Lot location or house number",'Proportion of main building',"Unit price (10,000 yuan/square meter)","City"],axis=1,inplace=True)
    dataset.drop(["parks_nearby_0_place_id"],axis=1,inplace=True)
    dataset.drop(["parks_nearby_1_place_id"],axis=1,inplace=True)
    dataset.drop(["parks_nearby_2_place_id"],axis=1,inplace=True)
    dataset.drop(["parks_nearby_3_place_id"],axis=1,inplace=True)
    dataset.drop(["parks_nearby_4_place_id"],axis=1,inplace=True)
    dataset.drop(["hospitals_nearby_0_place_id"],axis=1,inplace=True)
    dataset.drop(["hospitals_nearby_1_place_id"],axis=1,inplace=True)
    dataset.drop(["hospitals_nearby_2_place_id"],axis=1,inplace=True)
    dataset.drop(["hospitals_nearby_3_place_id"],axis=1,inplace=True)
    dataset.drop(["hospitals_nearby_4_place_id"],axis=1,inplace=True)
    dataset = dataset.fillna(0)
    #print(dataset.head(2).to_string())
    train = dataset[dataset["transaction date"]<"110/01/01"]
    test = dataset[dataset["transaction date"]>="110/01/01"]    
    train.drop("transaction date",axis=1,inplace=True)
    test.drop("transaction date",axis=1,inplace=True)

    percentages = []
    for fold_idx, (train_idx, valid_idx) in enumerate(ts_split.split(range(len(train)))):
        # print("Fold ID:", train_idx)
        # print("Fold ID:", valid_idx)
        mape = main2(train, train_idx, valid_idx)
        percentages.append(mape)
        print(percentages)    
    return np.mean(percentages)

if __name__ == '__main__':
    objective()