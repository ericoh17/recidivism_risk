import pandas as pd
import numpy as np

def calc_precision (t_range, test_dat):
  
  t_test_dat = test_dat.loc[test_dat['outcome_pred_all'] >= t_range]
  
  all_precision = ((1 - t_test_dat[['true_intervention']].values) \
                  / (1 - t_test_dat[['pred_intervention']].values)) \ 
                  * (t_test_dat[['true_outcome']].values \ 
                  - t_test_dat[['outcome_pred_ctrl']].values) \
                  + t_test_dat[['outcome_pred_ctrl']].values

  mean_precision = np.mean(all_precision)
  var_precision = np.var(all_precision)
  num_precision = t_test_dat.shape[0]

  precision_dict = {'precision' = mean_precision,
                    'precision_lower' = mean_precision \
                                        - 1.96 * np.sqrt(var_precision/num_precision),
                    'precision_upper' = mean_precision \
                                        + 1.96 * np.sqrt(var_precision/num_precision)
                    }

  return precision_dict


  
