import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta,timezone
import pdb
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

btc = pd.read_csv('BTC-2023-02_to_2026-02.csv', index_col=0)
fear_greed = pd.read_csv('fear-greed.csv', index_col=0)
btc = pd.concat([btc, fear_greed], axis = 1, join ='inner')
btc= btc.rename(columns ={"value": "fear_greed"}) ## do we need sentiment as a feature 
doge = pd.read_csv('DOGE-2023-02_to_2026-02.csv', index_col=0)
doge = pd.concat([doge, fear_greed], axis = 1, join ='inner')
doge= doge.rename(columns ={"value": "fear_greed"}) ## do we need sentiment as a feature 

features = [
    "volume_in_USDT",
    "funding_rate",
    "fear_greed"
]

target = ['30d_vol']

def LGBM_evaluation(df):
    X = df[features]
    y = df[target]

    # Time-series split (no shuffling)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1
    }

    model = lgb.train(
        params,
        train_data,
        valid_sets=[test_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=0)  # same as verbose_eval=False
        ]
    )


    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("Test RMSE:", rmse)

    importance = model.feature_importance(importance_type="gain")
    feature_names = model.feature_name()

    feat_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)

    return feat_imp

btc_ml = LGBM_evaluation(btc)
doge_ml = LGBM_evaluation(doge)
pdb.set_trace()
