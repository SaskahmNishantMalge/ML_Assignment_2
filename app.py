"""
Dry Bean Classifier – Streamlit front-end for ML Assignment 2.

Loads the five pipelines saved by notebooks/ModelTraining.ipynb and evaluates them
on a CSV the user uploads. Nothing is retrained here – the app only calls predict().

Run locally:  streamlit run app.py
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "model"
SAMPLE_CSV = APP_DIR / "test_data.csv"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "KNN": "knn",
    "Naive Bayes": "naive_bayes",
    "Support Vector Machine": "svm",
}
METRIC_ORDER = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

st.set_page_config(page_title="Dry Bean Classifier", page_icon="🫘", layout="wide")

st.markdown(
    """
    <style>
    /* Streamlit reserves 6rem above the page by default; trim it to just clear the 60px toolbar */
    .block-container, div[data-testid="stMainBlockContainer"] { padding-top: 4.25rem; }
    div[data-testid="stSidebarUserContent"] { padding-top: 1rem; }
    .bean-banner {
        background: linear-gradient(90deg, #4e6b2f 0%, #8a9a5b 100%);
        color: #fdfbf3; padding: 1.1rem 1.4rem; border-radius: 12px; margin-bottom: 1rem;
    }
    .bean-banner h1 { color: #fdfbf3; font-size: 1.9rem; margin: 0; }
    .bean-banner p  { margin: 0.25rem 0 0 0; opacity: 0.9; }
    div[data-testid="stMetric"] {
        background: rgba(138, 154, 91, 0.12); border: 1px solid rgba(78, 107, 47, 0.35);
        border-radius: 10px; padding: 0.6rem 0.8rem;
    }
    </style>
    <div class="bean-banner">
      <h1>🫘 Dry Bean Variety Classifier</h1>
      <p>Evaluate five pre-trained models on your own test split of the UCI Dry Bean dataset
      (16 shape features → 7 bean varieties).</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------- cached loaders ----------

@st.cache_resource
def load_models():
    return {name: joblib.load(MODEL_DIR / f"{key}.joblib") for name, key in MODEL_FILES.items()}


@st.cache_data
def load_feature_info():
    with open(MODEL_DIR / "feature_info.json") as f:
        return json.load(f)


@st.cache_data
def load_training_metrics():
    with open(MODEL_DIR / "metrics.json") as f:
        raw = json.load(f)
    return pd.DataFrame({v["name"]: v["test_metrics"] for v in raw.values()}).T[METRIC_ORDER]


# ---------- evaluation helpers ----------

def score_model(model, X, y_true):
    """Six assignment metrics; macro averages because the problem is multi-class."""
    y_pred = model.predict(X)
    proba = model.predict_proba(X)
    # Macro averages run over the varieties present in the file, so a file with only a
    # few varieties isn't penalised for the ones it doesn't contain.
    present = sorted(y_true.unique())
    # One-vs-rest macro AUC, written out so it also works when some classes are absent
    # (identical to roc_auc_score(..., multi_class="ovr") when all 7 are present).
    if len(present) > 1:
        class_pos = {c: i for i, c in enumerate(model.classes_)}
        auc = sum(roc_auc_score(y_true == c, proba[:, class_pos[c]]) for c in present) / len(present)
    else:
        auc = float("nan")
    scores = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, labels=present, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, labels=present, average="macro", zero_division=0),
        "F1": f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    return scores, y_pred


def plot_confusion(y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGn", cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax, linewidths=0.5)
    ax.set_xlabel("Predicted variety")
    ax.set_ylabel("True variety")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    fig.tight_layout()
    return fig


def validate_upload(df, features, target):
    """Return a list of blocking problems with the uploaded frame (empty list = OK)."""
    problems = []
    missing = [c for c in features if c not in df.columns]
    if missing:
        problems.append(f"Missing feature column(s): {', '.join(missing)}")
    if target not in df.columns:
        problems.append(f"Missing target column '{target}' – it is needed to compute metrics.")
    if not problems:
        non_numeric = [c for c in features if not pd.api.types.is_numeric_dtype(df[c])]
        if non_numeric:
            problems.append(f"Non-numeric values in: {', '.join(non_numeric)}")
        elif df[features + [target]].isna().any().any():
            problems.append("The file contains empty cells – please remove or fill them.")
    return problems


# ---------- sidebar ----------

feature_info = load_feature_info()
FEATURES, TARGET, CLASSES = feature_info["features"], feature_info["target"], feature_info["classes"]
models = load_models()

with st.sidebar:
    st.header("1 · Test data")
    uploaded = st.file_uploader("Upload a CSV with the 16 features + 'Class'", type="csv")
    if uploaded is None:
        st.caption("No file uploaded – showing results on the bundled test_data.csv.")
    if SAMPLE_CSV.exists():
        st.download_button("⬇️ Download sample test_data.csv", SAMPLE_CSV.read_bytes(),
                           file_name="test_data.csv", mime="text/csv")

    st.header("2 · Model")
    chosen_name = st.selectbox("Choose a classifier", list(MODEL_FILES), index=4)

    st.divider()
    st.caption("Models were trained once in the notebook on an 80/20 stratified split. "
               "This app only runs predictions – it never retrains.")

# ---------- load the evaluation data ----------

if uploaded is not None:
    eval_df = pd.read_csv(uploaded)
    source_label = uploaded.name
elif SAMPLE_CSV.exists():
    eval_df = pd.read_csv(SAMPLE_CSV)
    source_label = "bundled test_data.csv"
else:
    st.info("👈 Upload a test CSV in the sidebar to begin.")
    st.stop()

issues = validate_upload(eval_df, FEATURES, TARGET)
if issues:
    st.error("This file can't be evaluated:\n\n- " + "\n- ".join(issues))
    with st.expander("Expected columns"):
        st.write(FEATURES + [TARGET])
    st.stop()

unknown = sorted(set(eval_df[TARGET]) - set(CLASSES))
if unknown:
    st.warning(f"Dropping {eval_df[TARGET].isin(unknown).sum()} row(s) with unknown "
               f"class labels: {', '.join(map(str, unknown))}")
    eval_df = eval_df[~eval_df[TARGET].isin(unknown)]

X_eval = eval_df[FEATURES]
y_eval = eval_df[TARGET].astype(str)

st.caption(f"Evaluating on **{source_label}** · {len(eval_df):,} rows · "
           f"{y_eval.nunique()} of {len(CLASSES)} varieties present")

tab_eval, tab_compare, tab_preds, tab_about = st.tabs(
    ["📊 Model evaluation", "🏁 Compare all models", "🔍 Predictions", "ℹ️ About the data"]
)

# ---------- tab 1: single model ----------

with tab_eval:
    model = models[chosen_name]
    scores, y_hat = score_model(model, X_eval, y_eval)

    st.subheader(f"{chosen_name}")
    # two rows of three so the values stay readable on narrow screens
    cols = st.columns(3) + st.columns(3)
    for col, metric in zip(cols, METRIC_ORDER):
        value = scores[metric]
        col.metric(metric, "n/a" if pd.isna(value) else f"{value:.4f}",
                   help="AUC needs at least two varieties in the file" if metric == "AUC" else None)

    left, right = st.columns([1.1, 1])
    with left:
        st.pyplot(plot_confusion(y_eval, y_hat, CLASSES, f"Confusion matrix – {chosen_name}"))
    with right:
        st.markdown("**Classification report**")
        report = pd.DataFrame(
            classification_report(y_eval, y_hat, labels=CLASSES, output_dict=True, zero_division=0)
        ).T
        report["support"] = report["support"].astype(int)
        st.dataframe(report.style.format({"precision": "{:.3f}", "recall": "{:.3f}",
                                          "f1-score": "{:.3f}"})
                     .background_gradient(subset=["f1-score"], cmap="YlGn"),
                     width="stretch")
        wrong = int((y_hat != y_eval.values).sum())
        st.caption(f"{wrong} of {len(y_eval)} beans misclassified.")

# ---------- tab 2: all models side by side ----------

with tab_compare:
    st.subheader("All five models on this data")
    all_scores = pd.DataFrame(
        {name: score_model(m, X_eval, y_eval)[0] for name, m in models.items()}
    ).T[METRIC_ORDER]
    all_scores.index.name = "ML Model Name"
    st.dataframe(all_scores.style.format("{:.4f}").highlight_max(
                     props="background-color: #4e6b2f; color: #ffffff; font-weight: 600;"),
                 width="stretch")

    leader = all_scores["MCC"].idxmax()
    st.success(f"Best MCC on this file: **{leader}** ({all_scores.loc[leader, 'MCC']:.4f})")

    long_form = all_scores.reset_index().melt(id_vars="ML Model Name",
                                              var_name="Metric", value_name="Score")
    fig, ax = plt.subplots(figsize=(11, 4.2))
    sns.barplot(data=long_form, x="Metric", y="Score", hue="ML Model Name",
                palette="YlGn_r", ax=ax)
    ax.set_ylim(max(0.0, long_form["Score"].min() - 0.05), 1.0)
    ax.legend(loc="lower right", ncol=3, fontsize=8)
    ax.set_xlabel("")
    fig.tight_layout()
    st.pyplot(fig)

    with st.expander("Reference: scores recorded at training time (notebook)"):
        st.dataframe(load_training_metrics().style.format("{:.4f}"), width="stretch")

# ---------- tab 3: row-level predictions ----------

with tab_preds:
    st.subheader(f"Row-level predictions – {chosen_name}")
    out = eval_df.copy()
    out["Predicted"] = y_hat
    out["Correct"] = out["Predicted"] == y_eval.values
    proba = models[chosen_name].predict_proba(X_eval)
    out["Confidence"] = proba.max(axis=1).round(3)

    only_wrong = st.checkbox("Show only misclassified rows")
    view = out[~out["Correct"]] if only_wrong else out
    st.dataframe(view[[TARGET, "Predicted", "Confidence", "Correct"] + FEATURES],
                 width="stretch", height=380)
    st.download_button("⬇️ Download predictions CSV", out.to_csv(index=False).encode(),
                       file_name=f"predictions_{MODEL_FILES[chosen_name]}.csv", mime="text/csv")

# ---------- tab 4: dataset notes ----------

with tab_about:
    st.subheader("UCI Dry Bean Dataset")
    st.markdown(
        """
        - **Source:** UCI Machine Learning Repository (id 602) – Koklu & Ozkan, 2020.
        - **13,611 images** of dry beans from 7 registered Turkish varieties, reduced to
          **16 numeric shape features** (12 dimensional + 4 shape factors).
        - **Classes:** BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA (imbalanced –
          DERMASON ≈ 26 %, BOMBAY ≈ 4 %).
        - Metrics here are **macro-averaged**, so every variety counts equally; AUC is
          one-vs-rest.
        """
    )
    dist = y_eval.value_counts().reindex(CLASSES, fill_value=0)
    st.bar_chart(dist, color="#6b8e23")
