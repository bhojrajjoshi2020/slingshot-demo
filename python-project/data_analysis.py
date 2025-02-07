
import logging
import pandas as pd
import numpy as np
import glob

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from imblearn.over_sampling import SMOTE
import statsmodels.api as sm


logging.basicConfig(format='%(asctime)s %(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def execute_model(train_data):
    X = train_data.drop(["rampid", "group_flag", "avg_mcd_visit_binary", "campaign_initiative"], axis=1)
    y = train_data["group_flag"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logger.info("SMOTE fitting ...")
    smote = SMOTE(random_state=42)
    if len(train_data)<500:
        smote = SMOTE(random_state=42, k_neighbors=1)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    logger.info("Logistic Regression ...")
    rf = LogisticRegression()
    rf.fit(X_train_resampled, y_train_resampled)

    y_pred = rf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    logger.info("Completed model training!")

    logger.info("Calculating feature importance ...")
    coefficients = rf.coef_[0]
    indices = np.argsort(np.abs(coefficients))[::-1]
    feature_names = X.columns
    logger.info("Feature importance calculation complete!")

    logger.info("Starting weighted logistic regression ...")
    Z = scaler.fit_transform(X)
    probabilities = rf.predict_proba(Z)[:, 1]
    train_data["probabilities"] = probabilities
    train_data = train_data[(train_data["probabilities"] < 0.9999) & (train_data["probabilities"] > 0.0001)]
    train_data["weights"] = np.where(train_data["group_flag"] == 1, 1 / train_data["probabilities"], 1 / (1 - train_data["probabilities"]))

    X_wls = train_data[["group_flag"]]
    y_wls = train_data["avg_mcd_visit_binary"]
    weights_wls = train_data["weights"]
    X_wls = sm.add_constant(X_wls)
    model = sm.WLS(y_wls, X_wls, weights=weights_wls).fit()
    logger.info("Weighted logistic regression complete!")

    logger.info("Calculating odds ratio and confidence interval")
    odds_ratio = np.exp(model.params['group_flag'])
    conf_int = model.conf_int(alpha=0.20)
    L_CI = np.exp(conf_int[0][1])
    U_CI = np.exp(conf_int[1][1])
    logger.info("Odds ratio and confidence interval calculation complete!")

    return (train_data, feature_names, coefficients, indices, round(odds_ratio, 4), round(L_CI, 4), round(U_CI, 4))

def read_input(data_format, filepath):
    files = glob.glob(f'{filepath}')
    
    dfs = []
    for file in files:
        if file.endswith('_SUCCESS'):
            logger.info(f"Ignored success file -> {file}")
            continue
        logger.info(f'datafile -> {file}')
        logger.info(f'datafile -> {file}')
        if data_format.upper() == 'CSV':
           dfs.append(pd.read_csv(file))
        elif data_format.upper() == 'PARQUET':
           dfs.append(pd.read_parquet(file))
        else:
            raise Exception('Acceptable input data formats include csv or parquet only!')

    return pd.concat(dfs, ignore_index=True)

if __name__ == "__main__":
    
    traindata_df = read_input('CSV', '/Users/shynagen/py_ws/slingshot/*.csv')
    
    logger.info(traindata_df.size)
    
    traindata, features, coefficients, indices, x, y, z = execute_model(traindata_df)
