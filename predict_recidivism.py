#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3.6

# imports                                                                     
import numpy as np
import pandas as pd
import sqlite3
import lightgbm as lgb
import time
import os
import logging
from sklearn.linear_model import LogisticRegression
import seaborn

# imports for model selection and eval
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import classification_report
import eli5

# imports for model testing
from model_train_test.lgb_test import lgb_predict, lgb_feature_importance

# import functions for logging
from logger.logger import create_logger

# imports function to create features
from feature_engineering.feature_pipeline import create_features

# import counterfactual metrics functions
import DR_counterfactuals.calc_counterfactual_metrics as CM

# import observational metrics functions
import obs_metrics.calc_observational_metrics as OM

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
recidivism_train = pd.read_sql_query("SELECT id, decile_score, is_recid "
                                     "FROM recidivism_train ORDER BY id ASC", 
                                     db_conn)

Y_train = recidivism_train[['is_recid']]
score_train = recidivism_train[['decile_score']]
X_train = recidivism_train[['id']]
logger.info(f'Input training data has shape: {X_train.shape}')

# read in testing data
recidivism_test = pd.read_sql_query("SELECT id, decile_score, is_recid "
                                    "FROM recidivism_test ORDER BY id ASC", 
                                    db_conn)

Y_test = recidivism_test[['is_recid']]
score_test = recidivism_test[['decile_score']]
X_test = recidivism_test[['id']]
logger.info(f'Input testing data has shape: {X_test.shape}')

# create intervention variable
def score_to_intervention (row):
  if row['decile_score'] in {1, 2, 3, 4, 5}:
    return 0
  if row['decile_score'] in {6, 7, 8, 9, 10}:
    return 1

# convert each decile score to a binary variable
# denoting whether each subject received
# the intervention: 'extra' attention after release
train_intervention = score_train.apply(lambda row: score_to_intervention(row), \
                                                axis = 1)
score_train = score_train.assign(intervention = train_intervention.values)

test_intervention = score_test.apply(lambda row: score_to_intervention(row), \
                                              axis = 1)
score_test = score_test.assign(intervention = test_intervention.values)

# do feature engineering
logger.info("\nFeature engineering:\n")

X_train, X_test, _, _ = create_features(X_train, X_test, db_conn, cache_file)

logger.info(f'After feature engineering input training data has shape: {X_train.shape}')
logger.info(f'Training data columns: {X_train.columns.values.tolist()}\n')

logger.info(f'After feature engineering input testing data has shape: {X_test.shape}')
logger.info(f'Testing data columns: {X_test.columns.values.tolist()}')

# make categorical strings into type category
for col in ["sex", "race", "crime_degree", "custody_status", "marital_status"]:
  X_train[col] = X_train[col].astype('category')
  X_test[col] = X_train[col].astype('category')

X_train = X_train.drop(columns = ['id'])
X_test = X_test.drop(columns = ['id'])

# setup LightGBM parameters for CV
lgb_params = {
  "min_data_in_leaf": [10],
  "num_leaves": [30, 40, 50, 60],
  "max_depth": [3, 4, 5, 6],
  "learning_rate": [0.01, 0.05],
  "min_split_gain" : [0.01, 0.02],
#  "bagging_fraction": [0.3, 0.4, 0.5],
#  "bagging_freq": [2, 4, 6],
  "objective": ['binary']
}

n_folds = 10

# run CV
logger.info(f'\nCross Validation:\n')

outcome_classifier = lgb.LGBMClassifier()

outcome_cv = RandomizedSearchCV(outcome_classifier, 
                                param_distributions = lgb_params,
                                n_iter = 20,
                                cv = n_folds,
                                scoring = 'f1')

outcome_cv.fit(X_train, Y_train.values.ravel())

#logger.info(f'Best parameters from CV: {lgb_cv.best_params_}')
#logger.info(f'The mean CV ROC AUC: {lgb_cv.best_score_}')

# make predictions on test data using best params
#lgb_predict(lgb_cv, 
#            X_test.drop(columns = ['id']),
#            Y_test)

# one hot encode categorical features
#X_train_encode = pd.get_dummies(X_train, columns = ['sex', 'race', 'crime_degree',
#                                                    'custody_status', 'marital_status'],
#                                drop_first = True)

# observational outcome prediction model
#outcome_mod = LogisticRegression(random_state = 0,
#                                 solver = 'lbfgs').fit(X_train_encode.drop(columns = ['id']), 
#                                                       Y_train[['is_recid']].to_numpy)

obs_outcome_pred = outcome_cv.predict_proba(X_test)

# counterfactual outcome prediction model
cf_outcome_classifier = lgb.LGBMClassifier()

cf_outcome_cv = RandomizedSearchCV(cf_outcome_classifier,
                                   param_distributions = lgb_params,
                                   n_iter = 20,
                                   cv = n_folds,
                                   scoring = 'f1')

cf_outcome_cv.fit(X_train[score_train['intervention'] == 0],
                  Y_train[score_train['intervention'] == 0].values.ravel())

cf_outcome_pred = cf_outcome_cv.predict_proba(X_test)

# propensity prediction model
#propensity_mod = LogisticRegression(random_state = 0, 
#                                    solver = 'lbfgs').fit(X_train.drop(columns = ['id']), 
#                                                          score_train[['intervention']].to_numpy)

propensity_classifier = lgb.LGBMClassifier()

propensity_cv = RandomizedSearchCV(propensity_classifier,
                                   param_distributions = lgb_params,
                                   n_iter = 20,
                                   cv = n_folds,
                                   scoring = 'f1')

propensity_cv.fit(X_train,
                  score_train[['intervention']].values.ravel())

propensity_pred = propensity_cv.predict_proba(X_test)

# add predictions to test set
X_test['obs_outcome_pred'] = obs_outcome_pred[:,1]
X_test['cf_outcome_pred'] = cf_outcome_pred[:,1]
X_test['propensity_pred'] = propensity_pred[:,1]

# merge test dataset
full_test = pd.concat([Y_test, score_test['intervention'], X_test], axis = 1)

# calculate precision
#dr_precision = calc_precision(0.5, full_test)
#print(dr_precision)

# calculate recall
#dr_recall = calc_recall(0.5, full_test)
#print(dr_recall)

## get counterfactual metrics

# get PR curve
cf_PR_df = CM.calc_cf_PR_range(full_test)

# remove NAs
cf_PR_df = cf_PR_df.dropna()

# print PR curve
cf_PR_plot = seaborn.lineplot(x = 'recall',
                              y = 'precision',
                              hue = 'model',
                              data = cf_PR_df)
cf_PR_plot.set(ylim = (0, 1))
cf_PR_plot.set(xlim = (0, 1))
cf_PR_plot.set_title('Precision-Recall Curve with Counterfactual Evaluations')

cf_PR_plot.get_figure().savefig('cf_PR_plot.png')
cf_PR_plot.get_figure().clf()

# get ROC curve
cf_ROC_df = CM.calc_cf_ROC_range(full_test)

# remove NAs                                                                                                             
cf_ROC_df = cf_ROC_df.dropna()

# print PR curve                                                                                                                 
cf_ROC_plot = seaborn.lineplot(x = 'FPR',
                               y = 'recall',
                               hue = 'model',
                               data = cf_ROC_df)
cf_ROC_plot.set(ylim = (0, 1))
cf_ROC_plot.set(xlim = (0, 1))
cf_ROC_plot.set_title('ROC Curve with Counterfactual Evaluations')

cf_ROC_plot.get_figure().savefig('cf_ROC_plot.png')
cf_ROC_plot.get_figure().clf()

## get observational metrics

# get PR curve                                                                                                                  
obs_PR_df = OM.calc_obs_PR_range(full_test)                                                                                   

# remove NAs                                                                                                                      
obs_PR_df = obs_PR_df.dropna()                                                                                                      

# print PR curve                                                                                                                
obs_PR_plot = seaborn.lineplot(x = 'recall',                                                                                   
                               y = 'precision',                                                                              
                               hue = 'model',                                                                                        
                               data = obs_PR_df)                                                                             
obs_PR_plot.set(ylim = (0, 1))                                                                                                     
obs_PR_plot.set(xlim = (0, 1))
obs_PR_plot.set_title('Precision-Recall Curve with Observational Evaluations')                                                                                                 

obs_PR_plot.get_figure().savefig('obs_PR_plot.png')
obs_PR_plot.get_figure().clf()

# get ROC curve
obs_ROC_df = OM.calc_obs_ROC_range(full_test)

# remove NAs
obs_ROC_df = obs_ROC_df.dropna()

# print ROC curve
obs_ROC_plot = seaborn.lineplot(x = 'FPR',
                                y = 'recall',
                                hue = 'model',
                                data = obs_ROC_df)

obs_ROC_plot.set(ylim = (0, 1))
obs_ROC_plot.set(xlim = (0, 1))
obs_ROC_plot.set_title('ROC Curve with Observational Evaluations')

obs_ROC_plot.get_figure().savefig('obs_ROC_plot.png')
obs_ROC_plot.get_figure().clf()

# close sql connection
db_conn.close()

# shutdown logger
logging.shutdown()

