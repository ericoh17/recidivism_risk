import pandas as pd

def feat_justice(X_train, X_test, db_conn, cache_file):
  def _get_justice(df, table):
    query = f"""
    with temp_t AS (    
      SELECT
        id, 
        substr(name, 1, instr(name, ' ') - 1) AS FirstName,
        substr(name, instr(name, ' ') + 1) AS LastName
      FROM
        {table}
      )
    SELECT 
      t.id AS t_ID,
      t.LastName AS t_lastname,
      raw.LastName AS raw_lastname,
      t.FirstName AS t_firstname,
      raw.FirstName AS raw_firstname,
      raw.CustodyStatus AS custody_status,
      raw.MaritalStatus AS marital_status,
      raw.RecSupervisionLevel AS supervision_level
    FROM
      temp_t AS t
    LEFT JOIN
      "compas-scores-raw" AS raw
    ON
      LOWER(t_lastname) = LOWER(raw_lastname) AND LOWER(t_firstname) = LOWER(raw_firstname)
    ORDER BY
      t_id ASC      
    """

    temp_query_df = pd.read_sql_query(query, db_conn)

    df[[
      "custody_status",
      "marital_status",
      "supervision_level"
      ]] = temp_query_df[["custody_status", "marital_status", "supervision_level"]]
    #df = df.assign(custody_status = temp_query_df[['custody_status']].values,
    #               marital_status = temp_query_df[['marital_status']].values,
    #               supervision_level = temp_query_df[['supervision_level']].values)

  _get_justice(X_train, "recidivism_train")
  _get_justice(X_test, "recidivism_test")
  
  return X_train, X_test, db_conn, cache_file
