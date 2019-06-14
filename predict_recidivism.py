# imports                                                                     
import numpy as np
import pandas as pd
import sqlite3
import lightgbm as lgb
import time
import os

# imports for model selection and eval
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import classification_report
import eli5

# import functions for logging
from logger.logger import create_logger

# imports function to create features
from feature_engineering.feature_pipeline import create_features

np.random.seed(203)

# Logger
start_time = time.strftime("%Y-%m-%d_%H%M")
log_file = os.path.join("./log_output/", f'Run on {start_time}.log')
logger = create_logger(log_file)

# Globals                                                     
cache_file = './cache.db'
db_conn = sqlite3.connect("./data/recidivism_data.db")

# Optimize the db connection, don't forget to add the proper indexes as well          
db_conn('PRAGMA temp_store = MEMORY;')
db_conn(f'PRAGMA cache_size = {1 << 18};') # Page_size = 4096, Cache = 4096 * 2^18 = 1 0\73 741 824 Bytes

# read in training data
recidivism_train = pd.read_sql_query("SELECT id, is_recid FROM recidivism_train ORDER BY id ASC", db_conn)

Y_train = recidivism_train[['is_recid']]
X_train = recidivism_train[['id']]
logger.info(f'Input training data has shape: {X_train.shape}')

# read in testing data
recidivism_test = pd.read_sql_query("SELECT id, is_recid FROM recidivism_test ORDER BY id ASC", db_conn)

Y_test = recidivism_test[['is_recid']]
X_test = recidivism_test[['id']]
logger.info(f'Input testing data has shape: {X_test.shape}')

# do feature engineering
logger.info("Feature engineering")

X_train, X_test, _, _, _ = create_features(X_train, X_test, Y_train, db_conn, cache_file)

logger.info(f'After feature engineering input training data has shape: {X_train.shape}')
logger.info(f'Training data columns: {X_train.columns.values.tolist()}')

logger.info(f'After feature engineering input testing data has shape: {X_test.shape}')
logger.info(f'Testing data columns: {X_test.columns.values.tolist()}')

# make categorical strings into type category
for col in ["sex", "race", "crime_degree"]:
  X_train[col] = X_train[col].astype('category')
  X_test[col] = X_train[col].astype('category')

# setup LightGBM parameters for CV
lgb_params = {
  "num_leaves": [10, 12, 15],
  "max_depth": [3, 4, 5],
  "learning_rate": [0.1, 0.2, 0.3],
  "bagging_fraction": [0.3, 0.4, 0.5]
}

n_folds = 10

lgb_classifier = lgb.LGBMClassifier()

# run CV
lgb_cv = RandomizedSearchCV(lgb_classifier, 
                            param_distributions = lgb_params,
                            n_iter = 20,
                            cv = n_folds,
                            scoring = 'roc_auc')

lgb_cv.fit(X_train, Y_train.values.ravel())

logger.info(f'The mean CV ROC AUC: {lgb_cv.best_score_}')
logger.info(f'Weights of the best model: {eli5.show_weights(lgb_cv)}')

# make predictions on test data using best params
#Y_pred = lgb_cv.predict(X_test)

#print('Results on the test set:')
#print(classification_report(Y_test, Y_pred))

# close sql connection
db_conn.close()


