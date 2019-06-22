import pandas as pd

def feat_prev_crime(X_train, X_test, Y_train, db_conn, cache_file):
  def _get_prev_crime(df, table):
    query = f"""
    with temp_init_table AS (    
      SELECT
        id, 
        juv_fel_count,
        juv_misd_count,
        juv_other_count,
        priors_count
      FROM
        {table}
      ORDER BY
        id ASC
      )
    SELECT 
      juv_fel_count,
      juv_misd_count,
      juv_other_count,
      priors_count
    FROM temp_init_table      
    """

    df[[
      "juv_fel_count",
      "juv_misd_count",
      "juv_other_count",
      "priors_count"
      ]] = pd.read_sql_query(query, db_conn)

  _get_prev_crime(X_train, "recidivism_train")
  _get_prev_crime(X_test, "recidivism_test")
  
  return X_train, X_test, Y_train, db_conn, cache_file
