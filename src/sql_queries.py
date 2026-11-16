"""SQLite schema (normalised, 1:1 feature-group tables) and analytical queries."""
from __future__ import annotations

import sqlite3

import pandas as pd

from .data_utils import FAMILY_DESCRIPTIONS, ID_COL, TARGET, classify_columns, feature_family

FAMILIES = ["ind", "reg", "car", "calc"]


def _sql_type(col: str, df: pd.DataFrame) -> str:
    kind = classify_columns([col])
    if kind["bin"] or kind["cat"]:
        return "INTEGER"
    return "REAL" if str(df[col].dtype).startswith("float") and not col.endswith("_bin") else "INTEGER"


def family_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    feats = [c for c in df.columns if c not in (ID_COL, TARGET)]
    return {fam: [c for c in feats if feature_family(c) == fam] for fam in FAMILIES}


def build_schema(df: pd.DataFrame) -> list[str]:
    """DDL statements: policyholder + 4 feature-group tables + dictionary tables."""
    ddl = [
        "PRAGMA foreign_keys = ON;",
        f"CREATE TABLE policyholder ({ID_COL} INTEGER PRIMARY KEY, "
        f"{TARGET} INTEGER NOT NULL CHECK ({TARGET} IN (0,1)));",
        "CREATE TABLE feature_family (family TEXT PRIMARY KEY, description TEXT NOT NULL);",
        "CREATE TABLE feature_dictionary (feature TEXT PRIMARY KEY, "
        "family TEXT NOT NULL REFERENCES feature_family(family), kind TEXT, n_missing INTEGER, "
        "pct_missing REAL, n_unique INTEGER, univariate_gini REAL, chi2_pvalue REAL);",
    ]
    for fam, cols in family_columns(df).items():
        body = ", ".join(f"{c} {_sql_type(c, df)}" for c in cols)
        ddl.append(f"CREATE TABLE {fam}_features ({ID_COL} INTEGER PRIMARY KEY "
                   f"REFERENCES policyholder({ID_COL}), {body});")
    return ddl


def load_database(df: pd.DataFrame, feature_dict: pd.DataFrame, db_path, chunksize: int = 50_000):
    """Create the schema and bulk-load the cleaned data. Returns an open connection."""
    import os
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    for stmt in build_schema(df):
        conn.execute(stmt)
    df[[ID_COL, TARGET]].to_sql("policyholder", conn, if_exists="append", index=False, chunksize=chunksize)
    pd.DataFrame({"family": list(FAMILY_DESCRIPTIONS), "description": list(FAMILY_DESCRIPTIONS.values())}
                 ).query("family != 'other'").to_sql("feature_family", conn, if_exists="append", index=False)
    feature_dict.to_sql("feature_dictionary", conn, if_exists="append", index=False)
    for fam, cols in family_columns(df).items():
        df[[ID_COL] + cols].to_sql(f"{fam}_features", conn, if_exists="append", index=False, chunksize=chunksize)
    conn.commit()
    conn.execute("CREATE INDEX idx_ph_target ON policyholder(target);")
    conn.commit()
    return conn


# ----------------------------------------------------------------------------
# Analytical queries (10): JOIN, GROUP BY/HAVING, subquery, CTE, window functions
# ----------------------------------------------------------------------------
QUERIES: dict[str, tuple[str, str]] = {
"Q1 portfolio summary": ("aggregate", """
SELECT COUNT(*) AS policyholders, SUM(target) AS claims,
       ROUND(100.0 * AVG(target), 3) AS claim_rate_pct
FROM policyholder;"""),

"Q2 claim rate by vehicle category (JOIN + HAVING)": ("join, group by, having", """
SELECT c.ps_car_01_cat AS car_01_cat, COUNT(*) AS n, SUM(p.target) AS claims,
       ROUND(100.0 * AVG(p.target), 2) AS claim_rate_pct
FROM policyholder p JOIN car_features c ON c.id = p.id
GROUP BY c.ps_car_01_cat
HAVING COUNT(*) >= 1000
ORDER BY claim_rate_pct DESC;"""),

"Q3 driver x vehicle segments (3-table JOIN)": ("3-table join, having", """
SELECT i.ps_ind_17_bin AS ind_17, c.ps_car_02_cat AS car_02_cat, COUNT(*) AS n,
       ROUND(100.0 * AVG(p.target), 2) AS claim_rate_pct
FROM policyholder p
JOIN ind_features i ON i.id = p.id
JOIN car_features c ON c.id = p.id
GROUP BY i.ps_ind_17_bin, c.ps_car_02_cat
HAVING COUNT(*) >= 2000
ORDER BY claim_rate_pct DESC;"""),

"Q4 effect of missingness (CTE)": ("cte, join", """
WITH flags AS (
    SELECT p.id, p.target,
           (r.ps_reg_03 IS NULL)  AS reg03_missing,
           (c.ps_car_14 IS NULL)  AS car14_missing,
           (c.ps_car_03_cat IS NULL) AS car03_missing
    FROM policyholder p
    JOIN reg_features r ON r.id = p.id
    JOIN car_features c ON c.id = p.id)
SELECT reg03_missing, car14_missing, car03_missing, COUNT(*) AS n,
       ROUND(100.0 * AVG(target), 2) AS claim_rate_pct
FROM flags GROUP BY reg03_missing, car14_missing, car03_missing
HAVING COUNT(*) >= 500
ORDER BY n DESC;"""),

"Q5 top ps_car_11_cat levels (CTE + RANK window)": ("cte, rank, window avg", """
WITH lvl AS (
    SELECT c.ps_car_11_cat AS level, COUNT(*) AS n, AVG(p.target) AS rate
    FROM policyholder p JOIN car_features c ON c.id = p.id
    GROUP BY c.ps_car_11_cat HAVING COUNT(*) >= 500),
ranked AS (
    SELECT level, n, ROUND(100.0 * rate, 2) AS claim_rate_pct,
           RANK() OVER (ORDER BY rate DESC) AS rnk,
           ROUND(100.0 * (rate - AVG(rate) OVER ()), 2) AS diff_vs_level_mean_pp
    FROM lvl)
SELECT * FROM ranked WHERE rnk <= 10 ORDER BY rnk;"""),

"Q6 ps_car_13 deciles and lift (NTILE + scalar subquery)": ("ntile, subquery", """
WITH scored AS (
    SELECT p.target, c.ps_car_13,
           NTILE(10) OVER (ORDER BY c.ps_car_13) AS decile
    FROM policyholder p JOIN car_features c ON c.id = p.id
    WHERE c.ps_car_13 IS NOT NULL)
SELECT decile, COUNT(*) AS n, ROUND(MIN(ps_car_13), 3) AS lo, ROUND(MAX(ps_car_13), 3) AS hi,
       ROUND(100.0 * AVG(target), 2) AS claim_rate_pct,
       ROUND(AVG(target) / (SELECT AVG(target) FROM policyholder), 2) AS lift
FROM scored GROUP BY decile ORDER BY decile;"""),

"Q7 ps_ind_03 trend (LAG window)": ("cte, lag", """
WITH by_level AS (
    SELECT i.ps_ind_03 AS level, COUNT(*) AS n, AVG(p.target) AS rate
    FROM policyholder p JOIN ind_features i ON i.id = p.id
    GROUP BY i.ps_ind_03)
SELECT level, n, ROUND(100.0 * rate, 2) AS claim_rate_pct,
       ROUND(100.0 * (rate - LAG(rate) OVER (ORDER BY level)), 2) AS change_vs_prev_pp
FROM by_level ORDER BY level;"""),

"Q8 cumulative gain of ps_car_13 (running SUM window)": ("cte, cumulative window", """
WITH d AS (
    SELECT p.target, NTILE(10) OVER (ORDER BY c.ps_car_13 DESC) AS decile
    FROM policyholder p JOIN car_features c ON c.id = p.id
    WHERE c.ps_car_13 IS NOT NULL),
agg AS (SELECT decile, COUNT(*) AS n, SUM(target) AS claims FROM d GROUP BY decile)
SELECT decile, n, claims,
       ROUND(100.0 * SUM(claims) OVER (ORDER BY decile) / SUM(claims) OVER (), 2) AS cum_claims_captured_pct
FROM agg ORDER BY decile;"""),

"Q9 high-risk segments vs portfolio (HAVING + subquery)": ("3-table join, subquery in having", """
SELECT c.ps_car_01_cat AS car_01_cat, i.ps_ind_04_cat AS ind_04_cat, COUNT(*) AS n,
       ROUND(100.0 * AVG(p.target), 2) AS claim_rate_pct
FROM policyholder p
JOIN car_features c ON c.id = p.id
JOIN ind_features i ON i.id = p.id
GROUP BY c.ps_car_01_cat, i.ps_ind_04_cat
HAVING COUNT(*) >= 2000 AND AVG(p.target) > 1.15 * (SELECT AVG(target) FROM policyholder)
ORDER BY claim_rate_pct DESC LIMIT 10;"""),

"Q10 signal by feature family (dictionary JOIN)": ("join on dictionary", """
SELECT f.family, f.description, COUNT(*) AS n_features,
       SUM(d.n_missing > 0) AS features_with_missing,
       ROUND(AVG(ABS(d.univariate_gini)), 4) AS mean_abs_univariate_gini,
       ROUND(MAX(ABS(d.univariate_gini)), 4) AS max_abs_univariate_gini
FROM feature_dictionary d JOIN feature_family f ON f.family = d.family
GROUP BY f.family, f.description ORDER BY mean_abs_univariate_gini DESC;"""),
}


def run_query(conn: sqlite3.Connection, name: str) -> pd.DataFrame:
    """Run a named query and return the result as a DataFrame."""
    return pd.read_sql_query(QUERIES[name][1], conn)
