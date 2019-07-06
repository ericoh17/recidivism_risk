import pandas as pd

def feat_demographics(X_train, X_test, db_conn, cache_file):
  def _get_demographics(df, table):
    query = f"""
    with temp_init_table AS (    
      SELECT
        id, 
        sex,
        age,
        race
      FROM
        {table}
      ORDER BY
        id ASC
      )
    SELECT 
      sex,
      age,
      race
    FROM 
      temp_init_table      
    """

    df[[
      "sex",
      "age",
      "race"
      ]] = pd.read_sql_query(query, db_conn)

  _get_demographics(X_train, "recidivism_train")
  _get_demographics(X_test, "recidivism_test")
  
  return X_train, X_test, db_conn, cache_file
