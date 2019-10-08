import pandas as pd
import numpy as np

def calc_precision (t_range, test_dat):
  
  t_test_dat = test_dat[test_dat['obs_outcome_pred'] >= t_range]
  
  all_precision = (((1 - t_test_dat[['intervention']].values) \ 
                   (1 - t_test_dat[['propensity_pred']].values)) * 
                   (t_test_dat[['is_recid']].values - 
                   t_test_dat[['cf_outcome_pred']].values) + 
                   t_test_dat[['cf_outcome_pred']].values)

  mean_precision = np.mean(all_precision)
  var_precision = np.var(all_precision)
  num_precision = t_test_dat.shape[0]

  precision_dict = {'precision' = mean_precision,
                    'precision_lower' = (mean_precision -  
                                        1.96 * np.sqrt(var_precision/num_precision)),
                    'precision_upper' = (mean_precision + 
                                        1.96 * np.sqrt(var_precision/num_precision))
                    }

  return precision_dict


  
