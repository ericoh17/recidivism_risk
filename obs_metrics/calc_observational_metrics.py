import pandas as pd
import numpy as np

def calc_obs_precision (cutoff_val, test_dat, eval_type):

  if eval_type == 'obs':
      outcome_pred = test_dat['obs_outcome_pred'].values >= cutoff_val
  elif eval_type == 'cf':
      outcome_pred = test_dat['cf_outcome_pred'].values >= cutoff_val

  obs_precision = np.sum(outcome_pred * test_dat['is_recid'].values) / np.sum(outcome_pred)

  return obs_precision


def calc_obs_recall (cutoff_val, test_dat, eval_type):

  if eval_type == 'obs':
      outcome_pred = test_dat['obs_outcome_pred'].values >= cutoff_val
  elif eval_type == 'cf':
      outcome_pred = test_dat['cf_outcome_pred'].values >= cutoff_val

  obs_recall = np.sum(outcome_pred * test_dat['is_recid'].values) / np.sum(test_dat['is_recid'].values)

  return obs_recall


def calc_obs_fpr (cutoff_val, test_dat, eval_type):

  if eval_type == 'obs':
      outcome_pred = test_dat['obs_outcome_pred'].values >= cutoff_val
  elif eval_type == 'cf':
      outcome_pred = test_dat['cf_outcome_pred'].values >= cutoff_val

  true_negatives = 1 - test_dat['is_recid'].values
  obs_fpr = np.sum(outcome_pred * true_negatives) / np.sum(true_negatives)

  return obs_fpr


def calc_obs_PR_range (test_dat):

  cutoff_range = np.linspace(0, 1, 1001)

  obs_precision = [calc_obs_precision(c, test_dat, 'obs') for c in cutoff_range]

  obs_recall = [calc_obs_recall(c, test_dat, 'obs') for c in cutoff_range]

  cf_precision = [calc_obs_precision(c, test_dat, 'cf') for c in cutoff_range]

  cf_recall = [calc_obs_recall(c, test_dat, 'cf') for c in cutoff_range]

  obs_df = pd.DataFrame({'cutoff_range': cutoff_range,
                         'precision': obs_precision,
                         'recall': obs_recall,
                         'model': 'observational'})

  cf_df = pd.DataFrame({'cutoff_range': cutoff_range,
                        'precision': cf_precision,
                        'recall': cf_recall,
                        'model': 'counterfactual'})

  obs_PR_df = obs_df.append(cf_df, ignore_index = True)

  return obs_PR_df  


def calc_obs_ROC_range (test_dat):

  cutoff_range = np.linspace(0, 1, 1001)

  obs_recall = [calc_obs_recall(c, test_dat, 'obs') for c in cutoff_range]

  obs_fpr = [calc_obs_fpr(c, test_dat, 'obs') for c in cutoff_range]

  cf_recall = [calc_obs_recall(c, test_dat, 'cf') for c in cutoff_range]

  cf_fpr = [calc_obs_fpr(c, test_dat, 'cf') for c in cutoff_range]

  obs_df = pd.DataFrame({'cutoff_range': cutoff_range,
                         'recall': obs_recall,
                         'FPR': obs_fpr,
                         'model': 'observational'})

  cf_df = pd.DataFrame({'cutoff_range': cutoff_range,
                        'recall': cf_recall,
                        'FPR': cf_fpr,
                        'model': 'counterfactual'})

  obs_ROC_df = obs_df.append(cf_df, ignore_index = True)

  return obs_ROC_df
