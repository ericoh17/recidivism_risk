import pandas as pd
import numpy as np

def calc_precision (cutoff_val, test_dat):
  
  t_test_dat = test_dat[test_dat['obs_outcome_pred'] >= cutoff_val]
  
  all_precision = (((1 - t_test_dat[['intervention']].values) 
                   / (1 - t_test_dat[['propensity_pred']].values))
                   * (t_test_dat[['is_recid']].values
                   - t_test_dat[['cf_outcome_pred']].values)
                   + t_test_dat[['cf_outcome_pred']].values)

  mean_precision = np.mean(all_precision)
  var_precision = np.var(all_precision)
  num_precision = t_test_dat.shape[0]

  precision_dict = {'precision': mean_precision,
                    'precision_lower': (mean_precision
                                        - 1.96 * np.sqrt(var_precision/num_precision)),
                    'precision_upper': (mean_precision 
                                        + 1.96 * np.sqrt(var_precision/num_precision))
                    }

  return precision_dict


def calc_recall (cutoff_val, test_dat):

  outcome_pred = test_dat['obs_outcome_pred'].values >= cutoff_val

  outcome_estimate = (((1 - test_dat[['intervention']].values)
                      / (1 - test_dat[['propensity_pred']].values))
                      * (test_dat[['is_recid']].values
                      - test_dat[['cf_outcome_pred']].values)
                      + test_dat[['cf_outcome_pred']].values)

  recall = np.mean(outcome_pred * outcome_estimate) / np.mean(outcome_estimate)

  return recall


def calc_PR_df (test_dat):

  cutoff_range = np.linspace(0, 1, 1001)

  obs_precision = map(calc_precision, cutoff_range, test_dat)

  obs_recall = map(calc_recall, cutoff_range, test_dat)

  cf_precision = map(calc_precision, cutoff_range, test_dat)

  cf_recall = map(calc_recall, cutoff_range, test_dat)

  PR_df = pd.DataFrame({"cutoff_range": cutoff_range,
                        "obs_precision": obs_precision,
                        "obs_recall": obs_recall,
                        "cf_precision": cf_precision,
                        "cf_recall": cf_recall})

  return PR_df
