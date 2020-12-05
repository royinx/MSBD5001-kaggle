import numpy as np 
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import pickle
import holidays
SEARCH_PARAMS = {'learning_rate': 0.1424402189109501,
                'max_depth': 12,
                'num_leaves': 63,
                'num_trees':298, 
                'max_bin':482,
                'feature_fraction': 0.8602668445953969,
                'subsample': 0.805824598060849,
                }

def train_evaluate(search_params):
    folder_path = "msbd5001-fall2020/"
    train = pd.read_csv(f'{folder_path}train.csv')
    # LOAD TEST
    submit = pd.read_csv(f'{folder_path}test.csv')

    # shuffle
    train = train.sample(frac=1)
    train = train.reset_index(drop=True)

    # weather data 
    weather = pd.read_csv('hongkong_day.csv')
    # print(weather.loc[:100,["date_time"]])
    def parse_weather(dataset):
        dataset["date_time"] = pd.to_datetime(dataset["date_time"],format='%Y-%m-%d',)
        dataset = dataset.drop(['maxtempC', 'mintempC', 'totalSnow_cm', 'sunHour',
                                'uvIndex', 'moon_illumination', 'moonrise', 'moonset', 'sunrise',
                                'sunset', 'location'],
                                axis=1)
        # print(dataset.info())
        # exit()
        return dataset
    weather = parse_weather(weather)
    
    def parse(dataset):
        dataset["date"] = pd.to_datetime(dataset["date"],format='%d/%m/%Y %H:%M')
        dataset["date_"] = pd.to_datetime(dataset["date"].dt.date)
        dataset["year"] = dataset["date"].dt.year
        dataset["month"] = dataset["date"].dt.month
        dataset["quarter"] = dataset["date"].dt.quarter
        dataset["week_of_year"] = dataset["date"].dt.isocalendar().week.astype(int)
        dataset["day"] = dataset["date"].dt.day
        dataset["hour"] = dataset["date"].dt.hour

        dataset["weekday"] = dataset["date"].dt.weekday # Monday=0, Sunday=6.
        dataset['bl_weekend']=dataset['date'].apply(lambda x:0 if x.weekday() in [0,1,2,3,4] else 1)
        # dataset["isweekday"] = dataset["date"].dt.weekday # Monday=0, Sunday=6.
        dataset['day_of_year'] = dataset['date'].dt.dayofyear
        dataset['day_of_week'] = dataset['date'].dt.dayofweek

        ## is_holiday
        hk_holidays = holidays.HK()
        dataset['is_holiday'] = [ 1 if i in hk_holidays else 0 for i in dataset['date'] ]

        ## is_weekend
        dataset['is_weekend'] = [1 if i in [5,6] else 0 for i in dataset['weekday']]


        dataset = dataset.drop(['date'], axis=1)
        dataset = pd.get_dummies(dataset, prefix=['weekday'], columns=['weekday'])

        return dataset

    def prepare_dataset(X):
        # X["date"] = pd.to_datetime(X["date"])
        # print(X.info(), weather.info())
        # print(X[(X['date'] > '2017-01-01') & (X['date'] < '2017-01-02')])
        # print(weather[(weather['date_time'] > '2017-01-01') & (weather['date_time'] < '2017-01-02')])

        # X = train.loc[:,["date"]]
        X = parse(X.copy())
        X = pd.merge(X,weather, left_on="date_", right_on="date_time", how="left")

        X = X.drop(['date_','date_time'], axis=1)
        
        return X
    
    X = train.loc[:,["date"]]
    Y = train.loc[:,["speed"]]    
    
    # xtrain, xtest, ytrain, ytest = train_test_split(X,Y, test_size=0.1)
    xtrain = X 
    ytrain = Y
    # print(xtrain.head())
    # print(ytrain.head())
    print("========")
    # xtrain, xtest, ytrain, ytest = train_test_split(train.loc[:,["date"]], train["speed"], test_size=0.1)
    xtrain = prepare_dataset(xtrain)
    # xtest = prepare_dataset(xtest)
  
    # check data left join , testing*
    # xtrain.to_csv("lgb3_x.csv",index=False)
    # ytrain.to_csv("lgb3_y.csv",index=False)
    
    nfold = 10
    step = len(xtrain)//nfold
    score = 9999
    for idx in range(0,len(xtrain),step+1):
        x_cv_train = xtrain.copy().drop(xtrain.index[idx:idx+step+1])
        y_cv_train = ytrain.copy().drop(ytrain.index[idx:idx+step+1])
        x_cv_valid = xtrain.loc[idx:idx+step+1,:]
        y_cv_valid = ytrain.loc[idx:idx+step+1]
        # print(x_cv_train.shape,step+1)
        # """
        # print(dtrain.info())
        # exit()
        # ================================================
        dtrain = lgb.Dataset(x_cv_train,y_cv_train)

        params = {'objective': 'regression',
                'metric': 'rmse',
                'boosting':'gbdt',
                'objective':'regression', 
                'metric':'mse',
                "verbose":-1,
                #   'device_type':'gpu',
                **search_params}

        model = lgb.train(params, dtrain,
                        num_boost_round=300,
                        # early_stopping_rounds=30,
                        # valid_sets=[dtest],
                        # valid_names=['valid']
                        )
        # print(x_cv_valid.shape,y_cv_valid.shape)
        mse = mean_squared_error(model.predict(x_cv_valid), y_cv_valid)
        score = mse
        if score < 9.5:
            with open(f'best_model_{round(score,4)}.pkl', 'wb') as outfile:
                pickle.dump(model, outfile, protocol=pickle.HIGHEST_PROTOCOL)
            print(f'MSE updated: {mse}')
        # """
    # print(min(scores))
    # print(scores)
    # score_cv = min(cv_results['l2-mean'])
    # return min(scores) #, score_cv

    #   ================== Plot ==================

    # if True:
    #     import matplotlib.pyplot as plt
    #     features = []
    #     for k,v in zip(list(xtrain.columns),model.feature_importance()):
    #         features.append((k,v))

    #     def takeSecond(elem):
    #         return elem[1]
    #     # sort list with key
    #     features.sort(key=takeSecond,reverse=True)
    #     # print list
    #     print('Sorted list:', features)

        
    #     print('Plotting feature importances...')
    #     ax = lgb.plot_importance(model)#, max_num_features=10)
    #     plt.savefig('lgbm_importances-01.png')

    #   ================== load model and submit result ==================
    with open(f'best_model_8.9899.pkl', 'rb') as infile:
    # with open(f'best_model_{round(score,4)}.pkl', 'rb') as infile:
        best_model = pickle.load(infile)
    
    id = submit.loc[:,["id"]]
    submit = submit.loc[:,["date"]]
    submit = prepare_dataset(submit)
    result = best_model.predict(submit)
    result = pd.DataFrame(result,columns=["speed"])
    concat_result = pd.concat([id,result],axis=1)
    concat_result.to_csv("submit.csv",index=False)
    #   ==================  ==================  ==================

    return score

if __name__ == "__main__":
    score = train_evaluate(SEARCH_PARAMS)
    print('validation rmse:', score)



# best parameters:  [0.030741337320739335, 14, 38, 416, 477, 0.6853062771918034, 0.10837262574189999]
# MSE - 8.314068974219449
# best parameters:  [0.1424402189109501, 12, 63, 298, 482, 0.8602668445953969, 0.805824598060849]