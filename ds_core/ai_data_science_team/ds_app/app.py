

# # # # ----------------------------------------------------------------------------------------------------


from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Data Science Team",
    page_icon="images-logo/datascience_favicon.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# CSS
# =========================================================
st.markdown(
    """
<style>
    :root {
        --bg: #07090d;
        --panel: #10161f;
        --panel-2: #141b25;
        --border: rgba(255,255,255,0.08);
        --text: #f3f4f6;
        --muted: #98a2b3;
        --success-bg: rgba(20, 83, 45, 0.34);
        --success-border: rgba(74, 222, 128, 0.18);
        --danger-bg: rgba(127, 29, 29, 0.34);
        --danger-border: rgba(248, 113, 113, 0.18);
        --shadow: 0 14px 36px rgba(0,0,0,0.34);
    }

    html, body, [data-testid="stAppViewContainer"] {
        background: var(--bg);
        color: var(--text);
    }

    .intro-card {
        background: linear-gradient(180deg, rgba(16,22,31,0.96), rgba(10,14,20,0.96));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        box-shadow: var(--shadow);
        padding: 18px 20px;
        margin-bottom: 1rem;
    }

    .intro-title {
        font-size: 15px;
        font-weight: 700;
        margin-bottom: 6px;
        color: #f3f4f6 !important;
    }

    .intro-text {
        font-size: 14px;
        line-height: 1.7;
        color: #c8d0db !important;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(255,255,255,0.025), transparent 18%),
            linear-gradient(180deg, #05070b 0%, #07090d 100%);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        padding-top: 2.8rem;
        padding-bottom: 1.2rem;
        padding-left: 1.25rem;
        padding-right: 1.25rem;
        max-width: 1320px;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1017 0%, #090d14 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    [data-testid="stSidebar"][aria-expanded="false"] {
        min-width: 0 !important;
        max-width: 0 !important;
        width: 0 !important;
    }

    [data-testid="stSidebar"][aria-expanded="true"] {
        min-width: 330px !important;
        max-width: 330px !important;
    }

    [data-testid="stAppViewContainer"] > .main {
        transition: all 0.25s ease-in-out;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.25rem;
        padding-left: 0.9rem;
        padding-right: 0.9rem;
        padding-bottom: 1rem;
    }

    h1, h2, h3, h4, h5, h6, p, label, div, span {
        color: var(--text) !important;
    }

    .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-bottom: 1px solid rgba(255,255,255,0.06);
        padding-bottom: 1rem;
        margin-bottom: 1.35rem;
        gap: 14px;
    }

    .topbar-left {
        display: flex;
        align-items: center;
        gap: 14px;
        min-width: 0;
    }

    .logo-box {
        width: 46px;
        height: 46px;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, #1f2937, #111827);
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: var(--shadow);
        font-size: 19px;
        flex-shrink: 0;
    }

    .top-title {
        font-size: 20px;
        font-weight: 750;
        line-height: 1.15;
    }

    .top-subtitle {
        font-size: 13px;
        color: var(--muted) !important;
        line-height: 1.5;
        margin-top: 3px;
    }

    .deploy-pill {
        padding: 9px 16px;
        border-radius: 999px;
        background: linear-gradient(180deg, #171d27, #10151d);
        border: 1px solid rgba(255,255,255,0.08);
        font-size: 13px;
        font-weight: 650;
        white-space: nowrap;
        flex-shrink: 0;
    }

    .panel {
        background: linear-gradient(180deg, rgba(16,22,31,0.98), rgba(10,14,20,0.98));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        box-shadow: var(--shadow);
        padding: 16px 18px;
        overflow: hidden;
    }

    .panel-title {
        font-size: 15px;
        font-weight: 700;
        margin-bottom: 12px;
    }

    .sidebar-title {
        font-size: 15px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .small-muted {
        color: var(--muted) !important;
        font-size: 12px;
    }

    .metric-card {
        background: linear-gradient(180deg, #151b24, #10151d);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        box-shadow: var(--shadow);
        padding: 16px 10px;
        text-align: center;
        min-height: 92px;
    }

    .metric-value {
        font-size: 26px;
        font-weight: 750;
        margin-bottom: 4px;
    }

    .metric-label {
        font-size: 12px;
        color: var(--muted) !important;
    }

    .status-success {
        background: var(--success-bg);
        border: 1px solid var(--success-border);
        border-radius: 16px;
        padding: 14px 16px;
        margin-bottom: 16px;
    }

    .status-error {
        background: var(--danger-bg);
        border: 1px solid var(--danger-border);
        border-radius: 16px;
        padding: 14px 16px;
        margin-bottom: 16px;
    }

    .file-card {
        background: linear-gradient(180deg, #10161f, #0c1118);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 12px;
        box-shadow: var(--shadow);
        margin-top: 10px;
        margin-bottom: 12px;
    }

    .stTextInput input,
    .stTextArea textarea,
    .stNumberInput input,
    .stSelectbox div[data-baseweb="select"] > div,
    .stMultiSelect div[data-baseweb="select"] > div,
    .stRadio div[role="radiogroup"] {
        background: #121822 !important;
        color: var(--text) !important;
        border-radius: 14px !important;
    }

    .stFileUploader {
        background: linear-gradient(180deg, #10161f, #0c1118);
        border: 1px dashed rgba(255,255,255,0.12);
        border-radius: 18px;
        padding: 10px;
    }

    .stButton > button,
    .stDownloadButton > button {
        width: 100%;
        background: linear-gradient(180deg, #212937, #171d28);
        color: white;
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 14px;
        padding: 0.72rem 1rem;
        font-weight: 650;
        box-shadow: var(--shadow);
        transition: all 0.18s ease;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        transform: translateY(-1px);
        border-color: rgba(255,255,255,0.16);
    }

    .nav-current-pill {
        display: inline-block;
        padding: 0.32rem 0.7rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        color: #d7dfeb !important;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 0.9rem;
    }

    .ds-hero {
        position: relative;
        padding: 1rem 1rem 1.05rem 1rem;
        margin-bottom: 1rem;
        border-radius: 20px;
        background:
            radial-gradient(circle at top right, rgba(76, 201, 240, 0.16), transparent 30%),
            linear-gradient(180deg, rgba(12,16,23,0.96) 0%, rgba(8,11,17,0.98) 100%);
        border: 1px solid rgba(120, 190, 255, 0.14);
        box-shadow: 0 10px 28px rgba(0,0,0,0.22);
        overflow: hidden;
    }

    .ds-hero-badge {
        display: inline-block;
        padding: 0.22rem 0.55rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        color: #9ed8ff !important;
        background: rgba(79, 172, 254, 0.10);
        border: 1px solid rgba(79, 172, 254, 0.18);
        margin-bottom: 0.7rem;
    }

    .ds-hero-title {
        font-size: 1.35rem;
        font-weight: 800;
        line-height: 1.15;
        color: #f8fbff !important;
        margin-bottom: 0.45rem;
    }

    .ds-hero-subtitle {
        font-size: 0.92rem;
        line-height: 1.55;
        color: #b8c2cf !important;
    }

    .ds-overview-card {
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 20px;
        background: rgba(255,255,255,0.025);
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 20px rgba(0,0,0,0.18);
    }

    .ds-card-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #eef4ff !important;
        margin-bottom: 0.85rem;
    }

    .ds-stat-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.7rem;
    }

    .ds-stat-box {
        padding: 0.72rem 0.75rem;
        border-radius: 16px;
        background: linear-gradient(180deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.02) 100%);
        border: 1px solid rgba(255,255,255,0.06);
    }

    .ds-stat-label {
        font-size: 0.74rem;
        color: #8ea0b8 !important;
        margin-bottom: 0.22rem;
    }

    .ds-stat-value {
        font-size: 0.96rem;
        font-weight: 700;
        color: #f4f8ff !important;
    }

    .history-card {
        background: linear-gradient(180deg, #121822, #0d1219);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 12px 12px 10px 12px;
        box-shadow: var(--shadow);
        margin-bottom: 10px;
    }

    .history-role {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.04em;
        color: #9fb1c9 !important;
        text-transform: uppercase;
        margin-bottom: 6px;
    }

    .history-text {
        font-size: 13px;
        line-height: 1.55;
        color: #eef2f8 !important;
        word-break: break-word;
    }

    .analysis-grid-gap {
        height: 10px;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid rgba(255,255,255,0.06);
        margin-bottom: 1rem;
        overflow-x: auto;
        white-space: nowrap;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #c0c7d2;
        padding: 10px 16px;
        border-radius: 12px 12px 0 0;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background: #161d27 !important;
        color: white !important;
        border-bottom: 2px solid #f87171 !important;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.08);
        background: #0f141c;
    }

    [data-testid="stPlotlyChart"] {
        border-radius: 16px;
        overflow: hidden;
    }

    .markdown-answer p,
    .markdown-answer li,
    .markdown-answer strong,
    .markdown-answer code,
    .markdown-answer h1,
    .markdown-answer h2,
    .markdown-answer h3 {
        color: var(--text) !important;
    }

    @media (max-width: 900px) {
        [data-testid="stSidebar"][aria-expanded="true"] {
            min-width: 290px !important;
            max-width: 290px !important;
        }

        .topbar {
            flex-direction: column;
            align-items: flex-start;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# SESSION STATE
# =========================================================
def init_state():
    defaults = {
        "ds_page": "Chat Studio",
        "ds_df": None,
        "ds_source_name": None,
        "ds_chat_messages": [
            {
                "role": "assistant",
                "content": (
                    "Hello! I’m your Data Science Agent. If you need guidance with any dataset, just upload it and tell me what you’d like to do first."
                ),
            }
        ],
        "ds_openai_api_key": st.secrets.get("OPENAI_API_KEY", ""),
        "ds_openai_base_url": "https://api.openai.com/v1",
        "ds_openai_model": "gpt-4o-mini",
        "ds_preview_rows": 5,
        "ds_use_sample": False,
        "ds_last_provider_used": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# =========================================================
# DATA HELPERS
# =========================================================
def create_sample_df() -> pd.DataFrame:
    data = {
        "bike_model": [
            "Commuter Swift",
            "Urban Rider",
            "City Cruiser",
            "Mountain Trail Pro",
            "Gravel Explorer",
            "Road Velocity",
            "Eco E-Bike X",
            "Hybrid Motion",
            "Trail Beast",
        ],
        "bike_category": [
            "Commuter",
            "Urban",
            "City",
            "Mountain",
            "Gravel",
            "Road",
            "Electric",
            "Hybrid",
            "Mountain",
        ],
        "frame_material": [
            "Aluminum",
            "Aluminum",
            "Steel",
            "Aluminum",
            "Carbon",
            "Carbon",
            "Aluminum",
            "Steel",
            "Aluminum",
        ],
        "weight_kg": [11.8, 12.6, 13.9, 14.8, 10.2, 8.6, 19.7, 12.9, 15.3],
        "msrp_usd": [999, 799, 699, 1899, 2499, 2799, 3299, 1099, 2099],
        "is_electric": [0, 0, 0, 0, 0, 0, 1, 0, 0],
    }
    return pd.DataFrame(data)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [str(c).strip().lower().replace(" ", "_") for c in cleaned.columns]
    cleaned = cleaned.drop_duplicates()

    for col in cleaned.columns:
        if cleaned[col].dtype == object:
            cleaned[col] = cleaned[col].replace(["", "NA", "N/A", "null", "None"], np.nan)

    for col in cleaned.columns:
        if cleaned[col].dtype == object:
            converted = pd.to_numeric(cleaned[col], errors="coerce")
            if converted.notna().sum() >= max(3, len(cleaned) // 2):
                cleaned[col] = converted

    return cleaned


def dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(x) for x in df.dtypes],
            "missing": df.isna().sum().values,
            "unique": df.nunique(dropna=False).values,
        }
    )


def get_numeric_cols(df: pd.DataFrame):
    return df.select_dtypes(include=np.number).columns.tolist()


def get_categorical_cols(df: pd.DataFrame):
    return df.select_dtypes(exclude=np.number).columns.tolist()


def infer_target_candidate(df: pd.DataFrame) -> str:
    if "target" in df.columns:
        return "target"
    numeric_cols = get_numeric_cols(df)
    if numeric_cols:
        return numeric_cols[-1]
    return df.columns[-1] if len(df.columns) > 0 else ""


def build_dataset_context(df: Optional[pd.DataFrame]) -> str:
    if df is None:
        return "No dataset is currently loaded."

    preview = df.head(5).to_string(index=False)
    cols = ", ".join(df.columns.tolist())
    return f"""
Dataset shape: {df.shape}
Columns: {cols}

First 5 rows:
{preview}
""".strip()


# =========================================================
# REAL MODELLING HELPERS
# =========================================================
def _is_classification_target(series: pd.Series) -> bool:
    if series.dtype == object or str(series.dtype) == "category" or str(series.dtype) == "bool":
        return True

    non_null = series.dropna()
    if non_null.empty:
        return False

    unique_count = non_null.nunique()
    if unique_count <= 10:
        return True

    return False


def run_real_model(df: pd.DataFrame, target_col: str, model_name: str):
    data = clean_dataframe(df.copy())

    if target_col not in data.columns:
        raise ValueError("Selected target column not found in dataset.")

    data = data.dropna(subset=[target_col]).copy()

    if data.empty:
        raise ValueError("Dataset is empty after removing rows with missing target values.")

    X = data.drop(columns=[target_col]).copy()
    y = data[target_col].copy()

    if X.shape[1] == 0:
        raise ValueError("No feature columns available after removing the target column.")

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_transformer, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_transformer, categorical_cols))

    if not transformers:
        raise ValueError("No usable feature columns were found.")

    preprocessor = ColumnTransformer(transformers=transformers)

    is_classification = _is_classification_target(y)

    if is_classification:
        y = y.astype(str)
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        class_counts = pd.Series(y_encoded).value_counts()
        min_class_count = int(class_counts.min()) if not class_counts.empty else 0

        if len(np.unique(y_encoded)) < 2:
            raise ValueError("Classification requires at least 2 target classes.")

        if model_name == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        elif model_name == "Random Forest":
            model = RandomForestClassifier(random_state=42)
        elif model_name == "Gradient Boosting":
            model = GradientBoostingClassifier(random_state=42)
        else:
            model = RandomForestClassifier(random_state=42)

        stratify_value = y_encoded if min_class_count >= 2 else None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_encoded,
            test_size=0.2,
            random_state=42,
            stratify=stratify_value,
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        metrics = {
            "Task Type": "Classification",
            "Model": model.__class__.__name__,
            "Train Rows": int(len(X_train)),
            "Test Rows": int(len(X_test)),
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        }

        return metrics

    y_numeric = pd.to_numeric(y, errors="coerce")
    valid_idx = y_numeric.notna()

    X = X.loc[valid_idx].copy()
    y_numeric = y_numeric.loc[valid_idx].copy()

    if len(X) < 5:
        raise ValueError("Regression requires at least 5 valid rows after target cleaning.")

    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Random Forest":
        model = RandomForestRegressor(random_state=42)
    elif model_name == "Gradient Boosting":
        model = GradientBoostingRegressor(random_state=42)
    else:
        model = RandomForestRegressor(random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_numeric,
        test_size=0.2,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    metrics = {
        "Task Type": "Regression",
        "Model": model.__class__.__name__,
        "Train Rows": int(len(X_train)),
        "Test Rows": int(len(X_test)),
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": rmse,
        "R2 Score": r2_score(y_test, y_pred),
    }

    return metrics


# =========================================================
# NETWORK HELPERS
# =========================================================
def safe_request(
    url: str,
    headers: Optional[dict] = None,
    payload: Optional[dict] = None,
    timeout: int = 60,
):
    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


# =========================================================
# OPENAI ONLY
# =========================================================
def local_fallback_answer(prompt: str, df: Optional[pd.DataFrame]) -> str:
    if df is None:
        return (
            "## No dataset loaded\n\n"
            "Please upload a CSV first.\n\n"
            "After that, I can help with:\n"
            "- exploratory data analysis\n"
            "- data cleaning suggestions\n"
            "- feature engineering ideas\n"
            "- visualisation recommendations\n"
            "- modelling guidance"
        )

    return (
        f"## Agent response\n\n"
        f"You asked: **{prompt}**\n\n"
        f"The dataset currently contains **{df.shape[0]} rows** and **{df.shape[1]} columns**.\n\n"
        "### Recommended next step\n"
        "1. inspect missing values\n"
        "2. review numeric and categorical fields\n"
        "3. create visualisations\n"
        "4. identify a suitable target variable for modelling"
    )


def ask_openai(prompt: str, df: Optional[pd.DataFrame]) -> str:
    api_key = st.session_state.get("ds_openai_api_key") or st.secrets.get("OPENAI_API_KEY", "")
    api_key = api_key.strip() if api_key else ""

    base_url = st.session_state.ds_openai_base_url.strip().rstrip("/")
    model_name = st.session_state.ds_openai_model.strip()

    if not api_key:
        return "OpenAI API key is missing. Please configure it in Streamlit secrets."

    endpoint = f"{base_url}/chat/completions"
    context = build_dataset_context(df)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert data science agent inside a Streamlit application. "
                    "Respond in clear markdown. Give direct analysis based on the user's prompt and dataset context."
                ),
            },
            {
                "role": "user",
                "content": f"User prompt: {prompt}\n\nDataset context:\n{context}",
            },
        ],
        "temperature": 0.3,
    }

    try:
        data = safe_request(endpoint, headers=headers, payload=payload, timeout=90)
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"OpenAI request failed: {e}"


def run_agent(prompt: str, df: Optional[pd.DataFrame]) -> str:
    st.session_state.ds_last_provider_used = ""

    result = ask_openai(prompt, df)
    bad = result.lower().startswith("openai request failed:") or "api key is missing" in result.lower()

    if not bad:
        st.session_state.ds_last_provider_used = "OpenAI"
        return result

    st.session_state.ds_last_provider_used = "Local fallback"
    return local_fallback_answer(prompt, df) + f"\n\n---\n\n**OpenAI issue:** {result}"


# =========================================================
# UI HELPERS
# =========================================================
def render_topbar(page_title: str, subtitle: str, badge: str = "Deploy"):
    st.markdown(
        f"""
        <div class="topbar">
            <div class="topbar-left">
                <div class="logo-box"></div>
                <div>
                    <div class="top-title">{page_title}</div>
                    <div class="top-subtitle">{subtitle}</div>
                </div>
            </div>
            <div class="deploy-pill">{badge}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chat_messages():
    for msg in st.session_state.ds_chat_messages:
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            if msg["role"] == "assistant":
                st.markdown("<div class='markdown-answer'>", unsafe_allow_html=True)
                st.markdown(msg["content"])
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown(msg["content"])


# =========================================================
# SIDEBAR
# =========================================================
def render_sidebar():
    with st.sidebar:
        st.markdown("<div class='sidebar-title'>Navigation</div>", unsafe_allow_html=True)

        n1, n2, n3 = st.columns(3)
        with n1:
            if st.button("Chat", use_container_width=True, key="ds_nav_chat"):
                st.session_state.ds_page = "Chat Studio"
                st.rerun()
        with n2:
            if st.button("Explore", use_container_width=True, key="ds_nav_explore"):
                st.session_state.ds_page = "Data Explorer"
                st.rerun()
        with n3:
            if st.button("Model", use_container_width=True, key="ds_nav_model"):
                st.session_state.ds_page = "Modelling"
                st.rerun()

        st.markdown(
            f"<div class='nav-current-pill'>Current workspace: {st.session_state.ds_page}</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="ds-hero">
                <div class="ds-hero-badge">Data Science Agent</div>
                <div class="ds-hero-title">Data Science Workspace</div>
                <div class="ds-hero-subtitle">
                    Focused environment for dataset analysis, exploratory workflows, visual insights, and modelling execution.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="ds-overview-card">
                <div class="ds-card-title">Workspace Overview</div>
                <div class="ds-stat-grid">
                    <div class="ds-stat-box">
                        <div class="ds-stat-label">Mode</div>
                        <div class="ds-stat-value">Data</div>
                    </div>
                    <div class="ds-stat-box">
                        <div class="ds-stat-label">Stage</div>
                        <div class="ds-stat-value">{st.session_state.ds_page.replace("Data ", "")}</div>
                    </div>
                    <div class="ds-stat-box">
                        <div class="ds-stat-label">Dataset</div>
                        <div class="ds-stat-value">{"Ready" if st.session_state.ds_df is not None else "Waiting"}</div>
                    </div>
                    <div class="ds-stat-box">
                        <div class="ds-stat-label">Agent</div>
                        <div class="ds-stat-value">Active</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div class='sidebar-title'>Data options</div>", unsafe_allow_html=True)

        st.session_state.ds_use_sample = st.checkbox(
            "Load sample dataset",
            value=st.session_state.ds_use_sample,
        )
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="ds_upload_csv")
        st.session_state.ds_preview_rows = st.number_input(
            "Preview rows",
            min_value=3,
            max_value=50,
            value=int(st.session_state.ds_preview_rows),
            step=1,
        )

        if uploaded_file is not None:
            try:
                st.session_state.ds_df = pd.read_csv(uploaded_file)
                st.session_state.ds_source_name = uploaded_file.name
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")
        elif st.session_state.ds_use_sample and st.session_state.ds_df is None:
            st.session_state.ds_df = create_sample_df()
            st.session_state.ds_source_name = "sample_bike_dataset.csv"

        if st.session_state.ds_df is not None:
            file_size_kb = round(
                len(st.session_state.ds_df.to_csv(index=False).encode("utf-8")) / 1024,
                1,
            )
            st.markdown(
                f"""
                <div class="file-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; gap:10px;">
                        <div>
                            <div style="font-weight:700;">{st.session_state.ds_source_name}</div>
                            <div class="small-muted">{file_size_kb} KB • CSV</div>
                        </div>
                        <div class="small-muted">Loaded</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button("Delete current file", key="ds_delete_current_file"):
                st.session_state.ds_df = None
                st.session_state.ds_source_name = None
                st.session_state.ds_use_sample = False
                st.session_state.ds_last_provider_used = ""
                st.rerun()

        if st.session_state.ds_last_provider_used:
            st.markdown("<div class='sidebar-title'>Last response source</div>", unsafe_allow_html=True)
            st.caption(st.session_state.ds_last_provider_used)

        st.markdown("<div class='sidebar-title'>Chat history</div>", unsafe_allow_html=True)
        search_text = st.text_input("Search history", value="", key="ds_search_history")

        filtered = st.session_state.ds_chat_messages
        if search_text.strip():
            filtered = [
                m for m in st.session_state.ds_chat_messages
                if search_text.lower() in m["content"].lower()
            ]

        if filtered:
            for msg in filtered[-10:][::-1]:
                role = "User" if msg["role"] == "user" else "Assistant"
                text = msg["content"].replace("\n", " ")
                text = text[:120] + ("..." if len(text) > 120 else "")
                st.markdown(
                    f"""
                    <div class="history-card">
                        <div class="history-role">{role}</div>
                        <div class="history-text">{text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No matching chat history.")

        if st.button("Clear chat history", key="ds_clear_chat_history"):
            st.session_state.ds_chat_messages = []
            st.rerun()


# =========================================================
# PAGES
# =========================================================
def page_chat_studio():
    render_topbar(
        "Chat Studio",
        "Unified data science workspace for intelligent analysis, visual exploration, and modelling workflows",
        "Live",
    )

    st.markdown(
        """
    <div class="intro-card">
        <div class="intro-title">Welcome to the Data Science Team</div>
        <div class="intro-text">
            I can help with data cleaning, exploratory analysis, visualisation, feature engineering, and modelling workflows.
        </div>
        <div class="intro-text" style="margin-top: 10px;">
            <strong>Chat Studio:</strong> Interact with the agent, ask dataset-specific questions, and receive guided analysis support in conversation form.
        </div>
        <div class="intro-text" style="margin-top: 8px;">
            <strong>Explore:</strong> Analyse dataset structure, inspect cleaned data, review summaries, and generate interactive visual insights.
        </div>
        <div class="intro-text" style="margin-top: 8px;">
            <strong>Modelling:</strong> Prepare model inputs, configure the target workflow, and review pipeline execution with evaluation metrics.
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if st.session_state.ds_df is not None:
        st.markdown(
            "<div class='status-success'>Dataset loaded successfully. The agent can answer with dataset context.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='status-error'>No dataset loaded. You can still chat, but answers will be generic.</div>",
            unsafe_allow_html=True,
        )

    render_chat_messages()

    prompt = st.chat_input("Ask the data science team...")
    if prompt:
        st.session_state.ds_chat_messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Agent is typing..."):
                answer = run_agent(prompt, st.session_state.ds_df)
                st.markdown(answer)

        st.session_state.ds_chat_messages.append({"role": "assistant", "content": answer})
        st.rerun()


def page_data_explorer():
    render_topbar(
        "Data Explorer",
        "Interactive analysis details and visualisation workspace",
        "Analysis",
    )

    df = st.session_state.ds_df
    if df is None:
        st.markdown(
            "<div class='status-error'>No dataset loaded. Upload a CSV first from the sidebar.</div>",
            unsafe_allow_html=True,
        )
        return

    cleaned_df = clean_dataframe(df)
    engineered_df = cleaned_df.copy()

    if "msrp_usd" in engineered_df.columns and "weight_kg" in engineered_df.columns:
        engineered_df["price_per_kg"] = (
            engineered_df["msrp_usd"] / engineered_df["weight_kg"]
        ).round(2)

    with st.expander("Analysis Details", expanded=True):
        tabs = st.tabs(
            [
                "Data Ingestion",
                "Data Cleaning",
                "EDA",
                "Feature Engineering",
                "Visualisation",
                "Evaluation",
                "Modelling",
            ]
        )

        with tabs[0]:
            a, b, c, d = st.columns(4)
            with a:
                st.markdown(
                    f"<div class='metric-card'><div class='metric-value'>{df.shape[0]}</div><div class='metric-label'>Rows</div></div>",
                    unsafe_allow_html=True,
                )
            with b:
                st.markdown(
                    f"<div class='metric-card'><div class='metric-value'>{df.shape[1]}</div><div class='metric-label'>Columns</div></div>",
                    unsafe_allow_html=True,
                )
            with c:
                st.markdown(
                    f"<div class='metric-card'><div class='metric-value'>{int(df.isna().sum().sum())}</div><div class='metric-label'>Missing</div></div>",
                    unsafe_allow_html=True,
                )
            with d:
                st.markdown(
                    f"<div class='metric-card'><div class='metric-value'>{int(df.duplicated().sum())}</div><div class='metric-label'>Duplicates</div></div>",
                    unsafe_allow_html=True,
                )

            st.markdown("<div class='analysis-grid-gap'></div>", unsafe_allow_html=True)

            left, right = st.columns([1.35, 1.0], gap="large")
            with left:
                st.markdown("<div class='panel'><div class='panel-title'>Data preview</div>", unsafe_allow_html=True)
                st.dataframe(
                    df.head(int(st.session_state.ds_preview_rows)),
                    use_container_width=True,
                    height=400,
                )
                st.markdown("</div>", unsafe_allow_html=True)
            with right:
                st.markdown("<div class='panel'><div class='panel-title'>Schema profile</div>", unsafe_allow_html=True)
                st.dataframe(dataset_summary(df), use_container_width=True, height=400)
                st.markdown("</div>", unsafe_allow_html=True)

        with tabs[1]:
            l1, l2 = st.columns(2, gap="large")
            with l1:
                st.markdown("<div class='panel'><div class='panel-title'>Raw dataset</div>", unsafe_allow_html=True)
                st.dataframe(df.head(10), use_container_width=True, height=390)
                st.markdown("</div>", unsafe_allow_html=True)
            with l2:
                st.markdown("<div class='panel'><div class='panel-title'>Cleaned dataset</div>", unsafe_allow_html=True)
                st.dataframe(cleaned_df.head(10), use_container_width=True, height=390)
                st.markdown("</div>", unsafe_allow_html=True)

        with tabs[2]:
            l1, l2 = st.columns([1.45, 1.0], gap="large")
            with l1:
                st.markdown("<div class='panel'><div class='panel-title'>Summary statistics</div>", unsafe_allow_html=True)
                st.dataframe(
                    cleaned_df.describe(include="all").transpose(),
                    use_container_width=True,
                    height=450,
                )
                st.markdown("</div>", unsafe_allow_html=True)
            with l2:
                missing_df = cleaned_df.isna().sum().reset_index()
                missing_df.columns = ["column", "missing_values"]
                st.markdown("<div class='panel'><div class='panel-title'>Missing value profile</div>", unsafe_allow_html=True)
                st.dataframe(missing_df, use_container_width=True, height=450)
                st.markdown("</div>", unsafe_allow_html=True)

        with tabs[3]:
            st.markdown("<div class='panel'><div class='panel-title'>Engineered dataset</div>", unsafe_allow_html=True)
            st.dataframe(engineered_df.head(10), use_container_width=True, height=410)
            st.markdown("</div>", unsafe_allow_html=True)

        with tabs[4]:
            numeric_cols = get_numeric_cols(cleaned_df)
            categorical_cols = get_categorical_cols(cleaned_df)

            c1, c2, c3 = st.columns(3)
            with c1:
                chart_type = st.selectbox("Chart type", ["Histogram", "Bar", "Scatter", "Box", "Line"], key="ds_chart_type")
            with c2:
                x_axis = st.selectbox("X-axis", cleaned_df.columns.tolist(), key="ds_x_axis")
            with c3:
                y_axis = st.selectbox(
                    "Y-axis",
                    cleaned_df.columns.tolist(),
                    index=0 if not numeric_cols else cleaned_df.columns.tolist().index(numeric_cols[0]),
                    key="ds_y_axis",
                )

            if chart_type == "Histogram":
                fig = px.histogram(
                    cleaned_df,
                    x=x_axis,
                    template="plotly_dark",
                    height=660,
                    title=f"Distribution of {x_axis}",
                )
            elif chart_type == "Bar":
                if x_axis in categorical_cols:
                    bar_df = cleaned_df[x_axis].value_counts(dropna=False).reset_index()
                    bar_df.columns = [x_axis, "count"]
                    fig = px.bar(
                        bar_df,
                        x=x_axis,
                        y="count",
                        template="plotly_dark",
                        height=660,
                        title=f"Count of {x_axis}",
                    )
                else:
                    fig = px.bar(
                        cleaned_df.head(50),
                        x=x_axis,
                        y=y_axis,
                        template="plotly_dark",
                        height=660,
                        title=f"{y_axis} by {x_axis}",
                    )
            elif chart_type == "Scatter":
                fig = px.scatter(
                    cleaned_df,
                    x=x_axis,
                    y=y_axis,
                    template="plotly_dark",
                    height=660,
                    title=f"{y_axis} vs {x_axis}",
                )
            elif chart_type == "Box":
                fig = px.box(
                    cleaned_df,
                    x=x_axis,
                    y=y_axis,
                    template="plotly_dark",
                    height=660,
                    title=f"Box plot of {y_axis} by {x_axis}",
                )
            else:
                fig = px.line(
                    cleaned_df,
                    x=x_axis,
                    y=y_axis,
                    template="plotly_dark",
                    height=660,
                    title=f"{y_axis} over {x_axis}",
                )

            fig.update_layout(
                paper_bgcolor="#0f141c",
                plot_bgcolor="#0f141c",
                font=dict(color="#f3f4f6"),
                margin=dict(l=40, r=20, t=70, b=40),
            )

            st.markdown("<div class='panel'><div class='panel-title'>Visual output</div>", unsafe_allow_html=True)
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": True, "responsive": True, "scrollZoom": True},
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with tabs[5]:
            target_guess = infer_target_candidate(cleaned_df)
            st.markdown("<div class='panel'><div class='panel-title'>Evaluation summary</div>", unsafe_allow_html=True)
            st.write(f"Suggested target candidate: **{target_guess}**")
            st.write("Use this stage to review model readiness, leakage risk, and data quality.")
            st.markdown("</div>", unsafe_allow_html=True)

        with tabs[6]:
            st.markdown("<div class='panel'><div class='panel-title'>Modelling handoff</div>", unsafe_allow_html=True)
            st.write("Move to the Modelling page to run the real model workflow and evaluation metrics.")
            st.markdown("</div>", unsafe_allow_html=True)


def page_modelling():
    render_topbar(
        "Modelling Studio",
        "Premium modelling cards and results interface",
        "Model",
    )

    df = st.session_state.ds_df
    if df is None:
        st.markdown(
            "<div class='status-error'>No dataset loaded. Upload a CSV first from the sidebar.</div>",
            unsafe_allow_html=True,
        )
        return

    df = clean_dataframe(df)
    target_guess = infer_target_candidate(df)

    r1, r2, r3 = st.columns([1.2, 1.2, 0.8])

    with r1:
        selected_model = st.selectbox(
            "Select model",
            ["Linear Regression", "Logistic Regression", "Random Forest", "Gradient Boosting"],
            key="ds_select_model",
        )

    with r2:
        selected_target = st.selectbox(
            "Target column",
            df.columns.tolist(),
            index=df.columns.tolist().index(target_guess) if target_guess in df.columns else 0,
            key="ds_target_column",
        )

    with r3:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        run_model_btn = st.button("Run Model", key="ds_run_model")

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<div class='metric-card'><div class='metric-value'>Auto</div><div class='metric-label'>Feature Prep</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='metric-card'><div class='metric-value'>Ready</div><div class='metric-label'>Validation</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='metric-card'><div class='metric-value'>Live</div><div class='metric-label'>Metrics Panel</div></div>", unsafe_allow_html=True)

    if run_model_btn:
        try:
            with st.spinner("Training model and generating final report..."):
                result_metrics = run_real_model(df, selected_target, selected_model)

                steps = pd.DataFrame(
                    {
                        "step": ["Prepare data", "Train model", "Validate model", "Generate outputs"],
                        "status": ["Done", "Done", "Done", "Done"],
                    }
                )

                metrics = pd.DataFrame(
                    {
                        "metric": list(result_metrics.keys()),
                        "value": [
                            round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v
                            for v in result_metrics.values()
                        ],
                    }
                )

            st.markdown("<div class='status-success'>Model pipeline executed successfully.</div>", unsafe_allow_html=True)

            left, right = st.columns([1.15, 1.0], gap="large")
            with left:
                st.markdown("<div class='panel'><div class='panel-title'>Pipeline status</div>", unsafe_allow_html=True)
                st.dataframe(steps, use_container_width=True, height=260)
                st.markdown("</div>", unsafe_allow_html=True)

            with right:
                st.markdown("<div class='panel'><div class='panel-title'>Model metrics</div>", unsafe_allow_html=True)
                st.dataframe(metrics, use_container_width=True, height=260)
                st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.markdown(
                f"<div class='status-error'>Model training failed: {str(e)}</div>",
                unsafe_allow_html=True,
            )


# =========================================================
# MAIN
# =========================================================
def main():
    init_state()
    render_sidebar()

    if st.session_state.ds_page == "Chat Studio":
        page_chat_studio()
    elif st.session_state.ds_page == "Data Explorer":
        page_data_explorer()
    elif st.session_state.ds_page == "Modelling":
        page_modelling()
    else:
        st.session_state.ds_page = "Chat Studio"
        page_chat_studio()


if __name__ == "__main__":
    main()