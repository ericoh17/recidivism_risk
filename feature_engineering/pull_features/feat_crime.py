import pandas as pd

def feat_crime(X_train, X_test, db_conn, cache_file):
  def _get_crime(df, table):
    query = f"""
    with temp_init_table AS (    
      SELECT
        id, 
        c_charge_degree AS crime_degree,
        JulianDay(c_jail_out) - JulianDay(c_jail_in) AS jail_length,
        CASE(days_b_screening_arrest)
          WHEN days_b_screening_arrest < -30 AND days_b_screening_arrest > 30 THEN 0
          ELSE 1
        END screen_on_time
      FROM
        {table}
      ORDER BY
        id ASC
      )
    SELECT 
      crime_degree,
      jail_length,
      screen_on_time
    FROM 
      temp_init_table      
    """

    df[[
      "crime_degree",
      "jail_length",
      "screen_on_time"
      ]] = pd.read_sql_query(query, db_conn)

  _get_crime(X_train, "recidivism_train")
  _get_crime(X_test, "recidivism_test")
  
  return X_train, X_test, db_conn, cache_file
