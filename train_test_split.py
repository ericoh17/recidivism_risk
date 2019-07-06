import pandas as pd
import sqlite3

# Globals                                                                                                
cache_file = './cache.db'
db_conn = sqlite3.connect("./data/recidivism_data.db")

# Optimize the db connection, don't forget to add the proper indexes as well                        
db_conn('PRAGMA temp_store = MEMORY;')
db_conn(f'PRAGMA cache_size = {1 << 18};') # Page_size = 4096, Cache = 4096 * 2^18 = 1 0\73 741 824 Bytes  

# Import data
recidivism_dat = pd.read_sql_query("select * from `compas-scores-two-years`", db_conn)

# randomly sample 80% of data for training
# set the remaining 20% as test data
recidivism_train = recidivism_dat.sample(frac = 0.8, random_state = 1006)
recidivism_test = recidivism_dat.drop(recidivism_train.index)

recidivism_train.to_sql("recidivism_train", db_conn, if_exists = "replace")
recidivism_test.to_sql("recidivism_test", db_conn, if_exists = "replace")     
