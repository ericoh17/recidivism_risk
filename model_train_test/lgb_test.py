import pandas as pd
import numpy as np
import lightgbm as lgb
import logging
import shap
from sklearn.metrics import classification_report 

# get logger
logger = logging.getLogger("recidivism")

# make predictions on test data and 
# get feature importance using SHAP
def lgb_predict(cv_classifier, X_test, Y_test):
  
  Y_pred = cv_classifier.predict(X_test) 

  print('Results on the test set:\n')                                             
  print(classification_report(Y_test, Y_pred))     

  cv_params = cv_classifier.best_params_
  best_classifier = lgb.LGBMClassifier(num_leaves = cv_params['num_leaves'],
                                       max_depth = cv_params['max_depth'],
                                       learning_rate = cv_params['learning_rate'],
                                       bagging_fraction = cv_params['bagging_fraction'])
  best_classifier.fit(X_test, Y_test.values.ravel())
 
  lgb_feature_importance(best_classifier, X_test)

# get SHAP feature importance and plot  
def lgb_feature_importance(classifier, X_test):

  explainer = shap.TreeExplainer(classifier)
  shap_val = explainer.shap_values(X_test)

  shap.summary_plot(shap_val, X_test)
