import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

import holidays

folder_path = "msbd5001-fall2020/"
train = pd.read_csv(f'{folder_path}train.csv')
# LOAD TEST
test = pd.read_csv(f'{folder_path}test.csv')
print(train.shape, test.shape)

SEARCH_PARAMS = {'learning_rate': 0.4,
                'max_depth': 15,
                'num_leaves': 20,
                'feature_fraction': 0.8,
                'subsample': 0.2}

def train_evaluate(search_params):

    # weather data 
    weather = pd.read_csv('hongkong_hour.csv')
    # print(weather.loc[:100,["date_time"]])
    def parse_weather(dataset):
        dataset["date_time"] = pd.to_datetime(dataset["date_time"])
        dataset = dataset.drop(['maxtempC', 'mintempC', 'totalSnow_cm', 'sunHour',
                                'uvIndex', 'moon_illumination', 'moonrise', 'moonset', 'sunrise',
                                'sunset', 'winddirDegree', 'location'],
                                axis=1)
        # print(dataset.info())
        # exit()
        return dataset
    weather = parse_weather(weather)
    
    def parse(dataset):
        # dataset["date"] = pd.to_datetime(dataset["date"])

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


        dataset = dataset.drop(['date','date_time'], axis=1)
        dataset = pd.get_dummies(dataset, prefix=['weekday'], columns=['weekday'])

        return dataset
    #     dataset["weekday"] = dataset["date"].dt.()

    X = train.loc[:,["date"]]
    X["date"] = pd.to_datetime(X["date"])
    X = pd.merge(X,weather, left_on="date", right_on="date_time", how="left")
    # print(X[(X['date'] > '2017-01-01') & (X['date'] < '2017-01-02')])
    # print(weather[(weather['date_time'] > '2017-01-01') & (weather['date_time'] < '2017-01-02')])
    # print(X.info(), weather.info())
    X = parse(X)

    # X = X.loc[:,['hour',
    #             'cloudcover',
    #             'humidity',
    #             'day',
    #             'WindGustKmph',
    #             'day_of_year',
    #             'pressure',
    #             'windspeedKmph',
    #             'FeelsLikeC',
    #             'day_of_week',
    #             'DewPointC',
    #             'week_of_year',
    #             'tempC',
    #             'precipMM',
    #             'WindChillC',
    #             'HeatIndexC']]

    Y = train.loc[:,["speed"]]

    xtrain, xtest, ytrain, ytest = train_test_split(X,Y, test_size=0.1)
    
    # xtrain, xtest, ytrain, ytest = train_test_split(train.loc[:,["date"]], train["speed"], test_size=0.1)



    # ================================================
    dtrain = lgb.Dataset(xtrain,ytrain)
    dtest = lgb.Dataset(xtest,ytest,reference=dtrain)

    params = {'objective': 'regression',
              'metric': 'rmse',
              'boosting':'gbdt',
            #   'device_type':'gpu',
              **search_params}
    model = lgb.train(params, dtrain,
                    num_boost_round=300,
                    early_stopping_rounds=30,
                    valid_sets=[dtest],
                    valid_names=['valid'])
    score = model.best_score['valid']['rmse']

    # return score 


    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)



    # feature importance 
    if False:
        features = []
        for k,v in zip(list(xtrain.columns),model.feature_importance()):
            features.append((k,v))

        def takeSecond(elem):
            return elem[1]
        # sort list with key
        features.sort(key=takeSecond,reverse=True)
        # print list
        print('Sorted list:', features)

        
        print('Plotting feature importances...')
        ax = lgb.plot_importance(model)#, max_num_features=10)
        plt.savefig('lgbm_importances-01.png')

    return score 

if __name__ == "__main__":
    score = train_evaluate(SEARCH_PARAMS)
    print('validation rmse:', score)
