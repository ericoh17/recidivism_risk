import pandas as pd
import numpy as np

def calc_cf_precision (cutoff_val, test_dat, eval_type):

  if eval_type == 'obs':
    cut_test_dat = test_dat[test_dat['obs_outcome_pred'] >= cutoff_val]
  elif eval_type == 'cf':
    cut_test_dat = test_dat[test_dat['cf_outcome_pred'] >= cutoff_val]
  
  cut_precision = (((1 - cut_test_dat[['intervention']].values) 
                   / (1 - cut_test_dat[['propensity_pred']].values))
                   * (cut_test_dat[['is_recid']].values
                   - cut_test_dat[['cf_outcome_pred']].values)
                   + cut_test_dat[['cf_outcome_pred']].values)

  cf_precision = np.mean(cut_precision)
  
  return cf_precision


def calc_cf_recall (cutoff_val, test_dat, eval_type):
  
  if eval_type == 'obs':
      outcome_pred = test_dat['obs_outcome_pred'].values >= cutoff_val
  elif eval_type == 'cf':
      outcome_pred = test_dat['cf_outcome_pred'].values >= cutoff_val

  outcome_estimate = (((1 - test_dat[['intervention']].values)
                      / (1 - test_dat[['propensity_pred']].values))
                      * (test_dat[['is_recid']].values
                      - test_dat[['cf_outcome_pred']].values)
                      + test_dat[['cf_outcome_pred']].values)

  cf_recall = np.mean(outcome_pred * outcome_estimate) / np.mean(outcome_estimate)

  return cf_recall


def calc_cf_fpr (cutoff_val, test_dat, eval_type):
  
  if eval_type == 'obs':
      outcome_pred = test_dat['obs_outcome_pred'].values >= cutoff_val
  elif eval_type == 'cf':
      outcome_pred = test_dat['cf_outcome_pred'].values >= cutoff_val
  
  numerator_estimate = (((1 - test_dat[['intervention']].values)
                      / (1 - test_dat[['propensity_pred']].values))
                      * (test_dat[['cf_outcome_pred']].values
                      - test_dat[['is_recid']].values)
                      + (1 - test_dat[['cf_outcome_pred']].values))

  denominator_estimate = (((1 - test_dat[['intervention']].values)
                          / (1 - test_dat[['propensity_pred']].values))
                          * (test_dat[['is_recid']].values
                          - test_dat[['cf_outcome_pred']].values)
                          + test_dat[['cf_outcome_pred']].values)

  cf_fpr = np.mean(outcome_pred * numerator_estimate) / (1 - np.mean(denominator_estimate))

  return cf_fpr


def calc_cf_PR_range (test_dat):

  cutoff_range = np.linspace(0, 1, 1001)

  obs_precision = [calc_cf_precision(c, test_dat, 'obs') for c in cutoff_range]

  obs_recall = [calc_cf_recall(c, test_dat, 'obs') for c in cutoff_range]

  cf_precision = [calc_cf_precision(c, test_dat, 'cf') for c in cutoff_range]

  cf_recall = [calc_cf_recall(c, test_dat, 'cf') for c in cutoff_range]

  obs_df = pd.DataFrame({'cutoff_range': cutoff_range,
                         'precision': obs_precision,
                         'recall': obs_recall,
                         'type': 'observational'})
  
  cf_df = pd.DataFrame({'cutoff_range': cutoff_range,
                        'precision': cf_precision,
                        'recall': cf_recall,
                        'type': 'counterfactual'})

  cf_PR_df = obs_df.append(cf_df, ignore_index = True)

  return cf_PR_df


def calc_cf_ROC_range (test_dat):

  cutoff_range = np.linspace(0, 1, 1001)

  obs_recall = [calc_cf_recall(c, test_dat, 'obs') for c in cutoff_range]

  obs_fpr = [calc_cf_fpr(c, test_dat, 'obs') for c in cutoff_range]

  cf_recall = [calc_cf_recall(c, test_dat, 'cf') for c in cutoff_range]

  cf_fpr = [calc_cf_fpr(c, test_dat, 'cf') for c in cutoff_range]

  obs_df = pd.DataFrame({'cutoff_range': cutoff_range,
                         'recall': obs_recall,
                         'FPR': obs_fpr,
                         'type': 'observational'})

  cf_df = pd.DataFrame({'cutoff_range': cutoff_range,
                        'recall': cf_recall,
                        'FPR': cf_fpr,
                        'type': 'counterfactual'})

  cf_ROC_df = obs_df.append(cf_df, ignore_index = True)

  return cf_ROC_df
