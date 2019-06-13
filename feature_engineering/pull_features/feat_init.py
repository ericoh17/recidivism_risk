import pandas as pd

def feat_init(X_train, X_test, Y_train, db_conn, cache_file):
  def _get_init(df, table):
    query = f"""
    with temp_init_table AS (    
      SELECT
        id, 
        sex,
        age,
        race,
        priors_count,
        c_charge_degree AS crime_degree
      FROM
        {table}
      ORDER BY
        id ASC
      )
    SELECT 
      sex,
      age,
      race,
      priors_count,
      crime_degree
    FROM temp_init_table      
    """

    df[[
      "sex",
      "age",
      "race",
      "priors_count",
      "crime_degree"
      ]] = pd.read_sql_query(query, db_conn)

  _get_init(X_train, "recidivism_train")
  _get_init(X_test, "recidivism_test")
  
  return X_train, X_test, Y_train, db_conn, cache_file
