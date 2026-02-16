import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report


# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Dry Bean Multi-Class Classifier",
    page_icon="🫘",
    layout="wide"
)


# -----------------------------
# Utility Functions
# -----------------------------
@st.cache_resource
def load_assets():
    label_encoder = joblib.load("model/label_encoder.pkl")

    models = {
        "Logistic Regression": joblib.load("model/logistic_regression.pkl"),
        "Decision Tree": joblib.load("model/decision_tree.pkl"),
        "KNN": joblib.load("model/knn.pkl"),
        "Gaussian Naive Bayes": joblib.load("model/gaussian_nb.pkl"),
        "Random Forest (Ensemble)": joblib.load("model/random_forest.pkl"),
        "XGBoost (Ensemble)": joblib.load("model/xgboost.pkl"),
    }

    metrics_df = pd.read_csv("model/model_comparison_metrics.csv")

    return label_encoder, models, metrics_df


def validate_uploaded_data(df: pd.DataFrame, expected_features: list):
    """
    Ensures the uploaded CSV has exactly the required features.
    Order does not matter, but column names must match.
    """
    missing = [col for col in expected_features if col not in df.columns]
    extra = [col for col in df.columns if col not in expected_features]

    return missing, extra



def plot_confusion_matrix(cm: np.ndarray, class_names: np.ndarray, title: str):
    """
    Aesthetic confusion matrix visualization using pure matplotlib.
    No seaborn to keep dependencies minimal and deployment safe.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm)

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))

    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    # Annotate each cell with count
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


# -----------------------------
# Load Models + Encoders
# -----------------------------
label_encoder, models, metrics_df = load_assets()
class_names = label_encoder.classes_


# -----------------------------
# UI Layout
# -----------------------------
st.title("Dry Bean Multi-Class Classification App")
st.caption(
    "Upload a test CSV (features only), select a model, and get predictions + evaluation outputs."
)

col_left, col_right = st.columns([1.2, 1])

with col_left:
    st.subheader("1) Upload Test Dataset (CSV)")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

with col_right:
    st.subheader("2) Select Model")
    model_name = st.selectbox("Choose a trained model", list(models.keys()))
    selected_model = models[model_name]


st.divider()

st.subheader("Download Sample Test CSV")

with open("data/drybean_test_sample.csv", "rb") as f:
    st.download_button(
        label="Download Sample Test Data",
        data=f,
        file_name="drybean_test_sample.csv",
        mime="text/csv"
    )


# -----------------------------
# Show Model Comparison Metrics
# -----------------------------
st.subheader("Model Comparison Table (from training notebook)")
display_df = metrics_df.copy()

# Round numeric columns for better UI
for col in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]:
    if col in display_df.columns:
        display_df[col] = display_df[col].astype(float).round(4)

st.dataframe(display_df, use_container_width=True)


st.divider()


# -----------------------------
# Prediction Section
# -----------------------------
st.subheader(f"Predictions using: {model_name}")

if uploaded_file is None:
    st.info("Upload a CSV file to run predictions.")
    st.stop()

# Load uploaded data
test_df = pd.read_csv(uploaded_file)

# Determine expected feature columns from training dataset structure
# We assume the training dataset had all columns except 'Class'
# For safety, we infer expected features from the first model if it's a pipeline
expected_features = None

# Logistic/KNN/GNB are pipelines, trees/forests are not.
# We will infer expected columns from the uploaded CSV itself, but validate shape later.
# Better approach: store feature list during training. For now, enforce >= 12 and numeric.
if test_df.shape[1] < 12:
    st.error("Uploaded CSV must contain at least 12 feature columns as per assignment rules.")
    st.stop()

# Allow 'Class' column to be non-numeric (optional true labels)
allowed_non_numeric = ["Class"]

non_numeric_cols = test_df.select_dtypes(exclude=[np.number]).columns.tolist()
non_numeric_cols_filtered = [c for c in non_numeric_cols if c not in allowed_non_numeric]

if len(non_numeric_cols_filtered) > 0:
    st.error(
        f"Uploaded CSV contains non-numeric columns: {non_numeric_cols_filtered}. "
        "Please upload numeric feature columns only (except optional 'Class')."
    )
    st.stop()

X_test_upload = test_df.drop(columns=["Class"], errors="ignore")

# Predict
try:
    y_pred_upload = selected_model.predict(X_test_upload)
except Exception as e:
    st.error("Model failed to run predictions. Most likely column mismatch vs training features.")
    st.exception(e)
    st.stop()

# Convert predictions back to class names
predicted_labels = label_encoder.inverse_transform(y_pred_upload)

pred_out = pd.DataFrame({
    "Predicted_Class": predicted_labels
})

st.success(f"Predictions generated for {len(pred_out)} rows.")
st.dataframe(pred_out.head(20), use_container_width=True)


st.divider()


# -----------------------------
# Optional: Evaluate if user uploaded true labels
# -----------------------------
st.subheader("Evaluation on Uploaded Data (Optional)")

st.write(
    "If your uploaded CSV also contains a `Class` column (true labels), "
    "the app will compute confusion matrix and classification report."
)

if "Class" in test_df.columns:
    y_true_raw = test_df["Class"]

    # Encode true labels safely
    try:
        y_true = label_encoder.transform(y_true_raw)
    except Exception:
        st.error(
            "The `Class` column contains labels that do not match training classes. "
            "Ensure class names match exactly."
        )
        st.stop()

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred_upload)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Confusion Matrix")
        fig = plot_confusion_matrix(cm, class_names, f"Confusion Matrix — {model_name}")
        st.pyplot(fig)

    with col2:
        st.markdown("### Classification Report")
        report = classification_report(
            y_true,
            y_pred_upload,
            target_names=class_names,
            digits=4
        )
        st.text(report)

else:
    st.warning("No `Class` column found in uploaded CSV. Only predictions are shown.")