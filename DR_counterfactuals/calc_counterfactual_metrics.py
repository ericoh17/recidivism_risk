import pandas as pd
import numpy as np

def calc_precision (cutoff_val, test_dat, eval_type):

  if eval_type == 'obs':
    cut_test_dat = test_dat[test_dat['obs_outcome_pred'] >= cutoff_val]
  elif eval_type == 'cf':
    cut_test_dat = test_dat[test_dat['cf_outcome_pred'] >= cutoff_val]
  
  all_precision = (((1 - cut_test_dat[['intervention']].values) 
                   / (1 - cut_test_dat[['propensity_pred']].values))
                   * (cut_test_dat[['is_recid']].values
                   - cut_test_dat[['cf_outcome_pred']].values)
                   + cut_test_dat[['cf_outcome_pred']].values)

  mean_precision = np.mean(all_precision)
  #var_precision = np.var(all_precision)
  #num_precision = t_test_dat.shape[0]
  return mean_precision
  #precision_dict = {'precision': mean_precision,
  #                  'precision_lower': (mean_precision
  #                                      - 1.96 * np.sqrt(var_precision/num_precision)),
  #                  'precision_upper': (mean_precision 
  #                                      + 1.96 * np.sqrt(var_precision/num_precision))
  #                  }

  #return precision_dict


def calc_recall (cutoff_val, test_dat, eval_type):
  
  if eval_type == 'obs':
      outcome_pred = test_dat['obs_outcome_pred'].values >= cutoff_val
  elif eval_type == 'cf':
      outcome_pred = test_dat['cf_outcome_pred'].values >= cutoff_val

  outcome_estimate = (((1 - test_dat[['intervention']].values)
                      / (1 - test_dat[['propensity_pred']].values))
                      * (test_dat[['is_recid']].values
                      - test_dat[['cf_outcome_pred']].values)
                      + test_dat[['cf_outcome_pred']].values)

  recall = np.mean(outcome_pred * outcome_estimate) / np.mean(outcome_estimate)

  return recall


def calc_false_positive_rate (cutoff_val, test_dat, eval_type):
  
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

  fpr = np.mean(outcome_pred * numerator_estimate) / (1 - np.mean(denominator_estimate))

  return fpr


def calc_PR_range (test_dat):

  cutoff_range = np.linspace(0, 1, 1001)

  obs_precision = [calc_precision(c, test_dat, 'obs') for c in cutoff_range]

  obs_recall = [calc_recall(c, test_dat, 'obs') for c in cutoff_range]

  cf_precision = [calc_precision(c, test_dat, 'cf') for c in cutoff_range]

  cf_recall = [calc_recall(c, test_dat, 'cf') for c in cutoff_range]

  obs_df = pd.DataFrame({'cutoff_range': cutoff_range,
                         'precision': obs_precision,
                         'recall': obs_recall,
                         'type': 'observational'})
  
  cf_df = pd.DataFrame({'cutoff_range': cutoff_range,
                        'precision': cf_precision,
                        'recall': cf_recall,
                        'type': 'counterfactual'})

  PR_df = obs_df.append(cf_df, ignore_index = True)

  return PR_df


def calc_ROC_range (test_dat):

  cutoff_range = np.linspace(0, 1, 1001)

  obs_recall = [calc_recall(c, test_dat, 'obs') for c in cutoff_range]

  obs_fpr = [calc_false_positive_rate(c, test_dat, 'obs') for c in cutoff_range]

  cf_recall = [calc_recall(c, test_dat, 'cf') for c in cutoff_range]

  cf_fpr = [calc_false_positive_rate(c, test_dat, 'cf') for c in cutoff_range]

  obs_df = pd.DataFrame({'cutoff_range': cutoff_range,
                         'recall': obs_recall,
                         'FPR': obs_fpr,
                         'type': 'observational'})

  cf_df = pd.DataFrame({'cutoff_range': cutoff_range,
                        'recall': cf_recall,
                        'FPR': cf_fpr,
                        'type': 'counterfactual'})

  ROC_df = obs_df.append(cf_df, ignore_index = True)

  return ROC_df
