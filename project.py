import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta,timezone
import pdb
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

##============
## Load Data
##============

btc = pd.read_csv('BTC-2023-02_to_2026-02.csv', index_col=0)
fear_greed = pd.read_csv('fear-greed.csv', index_col=0)
btc = pd.concat([btc, fear_greed], axis = 1, join ='inner')
btc= btc.rename(columns ={"value": "fear_greed"}) ## do we need sentiment as a feature 
btc['target_30d_vol'] = btc['30d_vol'].shift(-30)
btc.dropna(inplace = True)
doge = pd.read_csv('DOGE-2023-02_to_2026-02.csv', index_col=0)
doge = pd.concat([doge, fear_greed], axis = 1, join ='inner')
doge= doge.rename(columns ={"value": "fear_greed"}) ## do we need sentiment as a feature 
doge['target_30d_vol'] = doge['30d_vol'].shift(-30)
doge.dropna(inplace = True)
##============================
## Setting features and target 
##============================

features = [
    "volume_in_USDT",
    "funding_rate",
    "fear_greed",
    '30d_vol'
]

target = ['target_30d_vol']

# def LGBM_evaluation(df):
#     X = df[features]
#     y = df[target]

#     # Time-series split (no shuffling)
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, shuffle=False
#     )

#     train_data = lgb.Dataset(X_train, label=y_train)
#     test_data = lgb.Dataset(X_test, label=y_test)

#     params = {
#         "objective": "regression",
#         "metric": "rmse",
#         "learning_rate": 0.05,
#         "num_leaves": 31,
#         "feature_fraction": 0.8,
#         "bagging_fraction": 0.8,
#         "bagging_freq": 5,
#         "verbose": -1
#     }

#     model = lgb.train(
#         params,
#         train_data,
#         valid_sets=[test_data],
#         callbacks=[
#             lgb.early_stopping(stopping_rounds=50),
#             lgb.log_evaluation(period=0)  # same as verbose_eval=False
#         ]
#     )


#     y_pred = model.predict(X_test)
#     rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#     print("Test RMSE:", rmse)

#     importance = model.feature_importance(importance_type="gain")
#     feature_names = model.feature_name()

#     feat_imp = pd.DataFrame({
#         "Feature": feature_names,
#         "Importance": importance
#     }).sort_values(by="Importance", ascending=False)

#     return feat_imp

# btc_ml = LGBM_evaluation(btc)
# doge_ml = LGBM_evaluation(doge)
# pdb.set_trace()

def run_models(df, name="Asset"):
    print(f"\n===== {name} =====")

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # ---------- LightGBM ----------
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

    lgb_model = lgb.train(
        params,
        train_data,
        valid_sets=[test_data],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(0)
        ]
    )

    y_pred_lgb = lgb_model.predict(X_test)
    rmse_lgb = np.sqrt(mean_squared_error(y_test, y_pred_lgb))
    r2_lgb = r2_score(y_test, y_pred_lgb)

    print(f"LightGBM RMSE: {rmse_lgb:.6f} | R²: {r2_lgb:.4f}")

    # ---------- LASSO ----------
    lasso_model = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", Lasso(alpha=0.001, max_iter=10000))
    ])

    lasso_model.fit(X_train, y_train)
    y_pred_lasso = lasso_model.predict(X_test)

    rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
    r2_lasso = r2_score(y_test, y_pred_lasso)

    print(f"LASSO RMSE:     {rmse_lasso:.6f} | R²: {r2_lasso:.4f}")

    lasso = lasso_model.named_steps["lasso"]
    coefs = lasso.coef_

    lasso_importance = pd.DataFrame({
        "Feature": features,
        "Coefficient": coefs,
        "Abs_Importance": np.abs(coefs)
    }).sort_values(by="Abs_Importance", ascending=False)

    print("\nLASSO Feature Importance:")
    print(lasso_importance)

    # ---------- Random Forest ----------
    rf_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )

    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    r2_rf = r2_score(y_test, y_pred_rf)

    print(f"RandomForest RMSE: {rmse_rf:.6f} | R²: {r2_rf:.4f}")

    rf_importance = pd.DataFrame({
        "Feature": features,
        "Importance": rf_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    print("\nRandom Forest Feature Importance:")
    print(rf_importance)

    # ---------- Feature Importance (LGBM) ----------
    importance = lgb_model.feature_importance(importance_type="gain")
    feat_imp = pd.DataFrame({
        "Feature": lgb_model.feature_name(),
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)

    print("\nTop Features (LightGBM):")
    print(feat_imp.head(5))

    # baseline RMSE to know performance 
    naive_pred = y_test.shift(1).bfill()  # yesterday's vol
    baseline_rmse = np.sqrt(mean_squared_error(y_test, naive_pred))

    print("Naive (yesterday vol) RMSE:", baseline_rmse)

    return feat_imp


btc_ml = run_models(btc, "BTC")
doge_ml = run_models(doge, "DOGE")

# pdb.set_trace()
