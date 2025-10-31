
"""
Colab-optimized utilities (patched from functions_for_LHS_w4.py)
- Safe optional imports (qgrid, seaborn)
- Robust feature name handling in ColumnTransformer
- Fixed bug in cohort_2_transform_df (num_varsitems -> num_vars.items)
- Avoid boolean masks as column selectors; use explicit lists
- qgrid fallback in Colab
- Optional CSV fallback for return_cohort (to avoid DB dependency)
"""

from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd
from urllib.parse import quote_plus as urlquote
try:
    from sqlalchemy import create_engine  # type: ignore
except Exception:
    create_engine = None  # will raise if DB path is used without SQLAlchemy

# Optional/soft imports that may not exist in Colab by default
try:
    import qgrid  # type: ignore
    _HAS_QGRID = True
except Exception:
    _HAS_QGRID = False

try:
    import seaborn as sns  # noqa: F401
except Exception:
    pass

# sklearn / imblearn
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline  # imblearn is required only for Pipeline here

# ---------- Connections & Data ----------

def return_cohort(username: str | None = None,
                  password: str | None = None,
                  cohort_type: int = 0,
                  csv_fallback_path: str | None = None) -> pd.DataFrame:
    """
    Retrieve cohort data either from Postgres or a CSV fallback.
    In Colab, prefer csv_fallback_path to avoid DB/credential issues.

    cohort_type:
      0 -> biogrid_vaed.RMH_COHORT
      1 -> biogrid_vaed.RMH_COHORT_WITHOUT_OUTLIERS
      2 -> biogrid_vaed.WHOLE_COHORT_WITHOUT_OUTLIERS
    """
    if csv_fallback_path and os.path.exists(csv_fallback_path):
        df = pd.read_csv(csv_fallback_path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # coerce types if present
            for col in ('visits_b2018', 'admissions_2017'):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
        return df

    # If no CSV fallback provided, attempt DB (not recommended in shared Colab sessions)
    POSTGRES_ADDRESS = 'alhs-data.validitron.io'
    POSTGRES_PORT = '5432'
    POSTGRES_DBNAME = 'lhscoursedb'

    if not username or not password:
        raise ValueError("username/password required for DB access or use csv_fallback_path.")

    postgres_str = (
        'postgresql://{username}:{password}@{ipaddress}:{port}/{dbname}'.format(
            username=username,
            password=urlquote(password),
            ipaddress=POSTGRES_ADDRESS,
            port=POSTGRES_PORT,
            dbname=POSTGRES_DBNAME
        )
    )

    if create_engine is None:
        raise ImportError('SQLAlchemy not available in this environment; use csv_fallback_path in Colab.')
    cnx = create_engine(postgres_str)
    if cohort_type == 0:
        sql = 'SELECT * FROM biogrid_vaed.RMH_COHORT;'
    elif cohort_type == 1:
        sql = 'SELECT * FROM biogrid_vaed.RMH_COHORT_WITHOUT_OUTLIERS;'
    elif cohort_type == 2:
        sql = 'SELECT * FROM biogrid_vaed.WHOLE_COHORT_WITHOUT_OUTLIERS;'
    else:
        raise ValueError("cohort_type must be 0, 1, or 2.")
    dataset = pd.read_sql_query(sql, cnx)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dataset = dataset.astype({'visits_b2018': 'float64', 'admissions_2017': 'float64'}, errors="ignore")
    return dataset


def dataframe_2_qgrid(df: pd.DataFrame):
    """
    Return a qgrid widget if available; otherwise return a lightweight preview
    with pandas styling (works in Colab). This keeps downstream code working.
    """
    if _HAS_QGRID:
        col_opts = {'editable': False}
        grid_options = {'forceFitColumns': False, 'defaultColumnWidth': 220, 'highlightSelectedCell': True}
        try:
            return qgrid.show_grid(df, column_options=col_opts, grid_options=grid_options)
        except Exception:
            pass  # fall through to styled preview
    # Fallback for Colab (no qgrid): return styled DataFrame (non-interactive)
    return (df.head(100)
            .style.set_table_attributes('style="display:inline"')
            .set_properties(**{'text-align': 'left'}))


# ----- Outlier helpers -----

def find_anomalies(data: np.ndarray | pd.Series):
    """Z-score based outlier detection for 1D data."""
    data = np.asarray(data, dtype=float)
    mu = np.nanmean(data)
    sigma = np.nanstd(data)
    cut = 3.0 * sigma
    lower, upper = mu - cut, mu + cut
    anomalies = [x for x in data if (x < lower) or (x > upper)]
    return anomalies, upper, lower


def find_outliers(df: pd.DataFrame, col_name: str):
    """IQR-based outliers for a DataFrame column."""
    s = pd.to_numeric(df[col_name], errors="coerce")
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    upper, lower = q3 + 1.5 * iqr, q1 - 1.5 * iqr
    anomalies = s[(s > upper) | (s < lower)].tolist()
    return anomalies, upper, lower


# ----- Feature descriptions (robust to missing keys) -----

col_name_2_discription = {}  # kept empty here; pass in your own mapping if needed

pyttype_2_vartype = {'object': 'Categorical', 'float64': 'Numerical', 'int64': 'Integer'}

def features_description(dataframe: pd.DataFrame, category: str = 'All',
                         col_desc: dict | None = None) -> pd.DataFrame:
    """
    Build a table of feature name/type/description. If a description is missing,
    use a generic string instead of raising.
    """
    if col_desc is None:
        col_desc = col_name_2_discription

    rows = []
    for feature in dataframe.columns:
        dtype = str(dataframe[feature].dtype)
        vartype = pyttype_2_vartype.get(dtype, dtype)
        descr = col_desc.get(feature, "(no description available)")
        rows.append((feature, vartype, descr))
    df = pd.DataFrame(rows, columns=['column name', 'type', 'description'])
    if category != 'All':
        return df[df['type'] == category].sort_values(by=['type', 'column name'], ignore_index=True)
    return df.sort_values(by=['type', 'column name'], ignore_index=True)


def label_2_name(data: pd.DataFrame, feature_name: str):
    new_name = f"{feature_name}_str"
    data.loc[(data[feature_name] == 1), new_name] = 'Admitted'
    data.loc[(data[feature_name] == 0), new_name] = 'Non-admitted'
    return data


# ----- Transform helpers -----

def cohort_2_transform_df(X_train: pd.DataFrame,
                          X_test: pd.DataFrame,
                          scaler: bool = True):
    """
    Fit a preprocessing ColumnTransformer on X_train and transform both splits.
    - Numeric: median impute (+ optional standardize)
    - Categorical: OneHotEncoder(handle_unknown='ignore')
    Returns (X_train_np, X_train_df, X_test_np, X_test_df)
    """
    # Explicit column lists (avoid boolean masks – more stable with ColumnTransformer)
    num_cols = X_train.select_dtypes(exclude=['object', 'category']).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    si_0 = SimpleImputer(missing_values=np.nan, strategy='median')
    ss = StandardScaler()
        # Handle scikit-learn version differences (sparse_output vs sparse)
    try:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

    categorical_pipe = Pipeline([('ohe', ohe)])
    numeric_pipe = Pipeline([('si_0', si_0)] + ([('ss', ss)] if scaler else []))

    col_transformer = ColumnTransformer(
        transformers=[
            ('nums', numeric_pipe, num_cols),
            ('cats', categorical_pipe, cat_cols)
        ],
        remainder='drop',
        n_jobs=None  # avoid parallelism issues on Colab
    )

    # Fit/transform
    X_train_np = col_transformer.fit_transform(X_train)
    X_test_np = col_transformer.transform(X_test)

    # Feature names
    names_num = num_cols
    try:
        ohe_names = col_transformer.named_transformers_['cats']['ohe'].get_feature_names_out(cat_cols).tolist()
    except Exception:
        # Fallback if feature names unavailable
        ohe_names = [f"cat_{i}" for i in range(X_train_np.shape[1] - len(names_num))]
    names_all = names_num + ohe_names

    X_train_df = pd.DataFrame(X_train_np, columns=names_all, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_np, columns=names_all, index=X_test.index)
    return X_train_np, X_train_df, X_test_np, X_test_df