
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.cluster import KMeans
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
import traceback

# Suppress warnings for cleaner display, especially during development
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Dashboard Prediksi Penyakit",
    page_icon="🩺",
)
st.title("Aplikasi Prediksi Penyakit Berdasarkan Gejala")
st.write("Masukkan data gejala pasien untuk memprediksi kemungkinan penyakit.")

# --- Define Colors for Plots (consistent with dark theme) ---
DARK_BG_COLOR = '#2E2E2E'
LIGHT_TEXT_COLOR = '#EAEAEA'
OUTLINE_COLOR = '#777777'
GRID_COLOR = '#555555'
HEATMAP_LINE_COLOR = '#444444'

def process_model_probabilities(prob_output, n_classes, n_samples=None):
    """
    Processes the raw probability output from a model to ensure it's in a consistent 2D format.
    Handles cases where predict_proba might return 1D for binary classification or single samples.
    """
    try:
        probs = np.array(prob_output)
        if probs.ndim == 2:
            return probs
        elif probs.ndim == 1 and n_samples == 1:
            # For a single prediction, if 1D and binary, reshape to [[prob_class_0, prob_class_1]]
            if n_classes == 2 and probs.shape[0] == 1:
                return np.array([[1 - probs[0], probs[0]]])
            # If 1D and multi-class (less common), reshape to [[prob1, prob2, ...]]
            elif probs.shape[0] == n_classes:
                return probs.reshape(1, -1)
            else: # Fallback if dimensions don't match
                return np.ones((1, n_classes)) / n_classes if n_classes > 0 else np.array([])
        else:
            # Fallback for unexpected dimensions, return uniform probabilities
            return np.ones((n_samples if n_samples else 1, n_classes)) / n_classes if n_classes > 0 else np.array([])
    except Exception:
        # Catch any error during processing and return uniform probabilities as a safe default
        return np.ones((n_samples if n_samples else 1, n_classes)) / n_classes if n_classes > 0 else np.array([])


def process_model_predictions(pred_output):
    """
    Processes raw prediction output to ensure it's a flat 1D numpy array of class labels.
    """
    try:
        pred_output = np.array(pred_output)
        if pred_output.ndim == 1:
            return pred_output
        elif pred_output.ndim == 2 and pred_output.shape[1] == 1:
            return pred_output.flatten()
        elif pred_output.ndim == 2 and pred_output.shape[1] > 1:
            # For multi-class (e.g., one-hot encoded predictions), take argmax
            return np.argmax(pred_output, axis=1)
        elif np.isscalar(pred_output):
            return np.array([pred_output])
        return pred_output.flatten() # Ensure it's 1D
    except Exception:
        return np.array([]) # Return empty array on error

@st.cache_resource
def load_resources():
    """
    Loads all necessary data files and the trained model.
    Uses st.cache_resource for efficient loading across app runs.
    """
    model_res, df_res, desc_df_res, precaution_df_res, symptom_severity_df_res = None, None, None, None, None
    all_loaded_successfully = True
    try:
        model_res = joblib.load('diseaseprediction.joblib')
        df_res = pd.read_csv('dataset.csv')
        desc_df_res = pd.read_csv('symptom_Description.csv')
        precaution_df_res = pd.read_csv('symptom_precaution.csv')
    except FileNotFoundError:
        st.error("Satu atau lebih file data inti tidak ditemukan (diseaseprediction.joblib, dataset.csv, symptom_Description.csv, symptom_precaution.csv). Aplikasi tidak dapat berjalan.")
        all_loaded_successfully = False
    except Exception as e:
        st.error(f"Gagal memuat sumber daya inti: {e}. Aplikasi tidak dapat berjalan.")
        all_loaded_successfully = False

    try:
        symptom_severity_df_res = pd.read_csv('Symptom-severity.csv')
        symptom_severity_df_res.columns = [col.strip().lower() for col in symptom_severity_df_res.columns]
        if 'symptom' not in symptom_severity_df_res.columns or 'weight' not in symptom_severity_df_res.columns:
            st.warning("File 'Symptom-severity.csv' tidak memiliki kolom 'symptom' atau 'weight'. Skor keparahan tidak dapat dihitung/digunakan.")
            symptom_severity_df_res = None
        else:
            symptom_severity_df_res['symptom'] = symptom_severity_df_res['symptom'].astype(str).str.strip().str.lower().str.replace(' ', '_')
    except FileNotFoundError:
        st.warning("File 'Symptom-severity.csv' tidak ditemukan. Skor keparahan tidak akan dihitung/digunakan oleh model atau ditampilkan.")
        symptom_severity_df_res = None
    except Exception as e:
        st.warning(f"Gagal memuat 'Symptom-severity.csv': {e}. Skor keparahan tidak akan dihitung/digunakan atau ditampilkan.")
        symptom_severity_df_res = None

    return model_res, df_res, desc_df_res, precaution_df_res, symptom_severity_df_res, all_loaded_successfully

# --- Load Resources ---
model, df_orig, desc_df, precaution_df, symptom_severity_df, load_success = load_resources()

if not load_success or model is None or df_orig is None or desc_df is None or precaution_df is None:
    st.error("Pemuatan sumber daya inti gagal. Aplikasi tidak dapat melanjutkan.")
    st.stop()

# --- Data Preprocessing (as done during model training) ---
# Create a copy to avoid modifying the original loaded DataFrame
df = df_orig.copy()

# Drop unnecessary symptom columns if they exist in the dataset
cols_to_drop = ['Symptom_12','Symptom_13','Symptom_14','Symptom_15','Symptom_16','Symptom_17']
for col in cols_to_drop:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

SYMPTOM_PLACEHOLDER = 'unknown_symptom_placeholder'
symptom_string_cols = [col for col in df.columns if 'Symptom' in col and col != 'severity_score']

# Normalize and encode string symptoms in the training data
for col in symptom_string_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.lower().str.replace(' ', '_')
        # Replace empty strings, 'nan', 'none', 'null' with a consistent placeholder
        df[col] = df[col].replace(['nan', '', 'none', 'null'], SYMPTOM_PLACEHOLDER)

# Prepare symptom encoder based on all possible symptom values encountered in the dataset
all_symptom_values_for_encoder = [SYMPTOM_PLACEHOLDER]
for col in symptom_string_cols:
    if col in df.columns:
        all_symptom_values_for_encoder.extend(df[col].unique())
unique_symptom_encoder_values = sorted(list(set(all_symptom_values_for_encoder)))
symptom_encoder = LabelEncoder()
symptom_encoder.fit(unique_symptom_encoder_values)
symptom_mapping = dict(zip(symptom_encoder.classes_, symptom_encoder.transform(symptom_encoder.classes_)))

# Encode symptoms to numerical values for the training data
for col in symptom_string_cols:
    if col in df.columns:
        df[col] = df[col].map(symptom_mapping).fillna(symptom_mapping.get(SYMPTOM_PLACEHOLDER, 0))

# Prepare list of symptoms for the Streamlit multiselect (using original string format for display)
symptoms_for_multiselect_set = set()
for col in symptom_string_cols:
    if col in df_orig.columns: # Use df_orig to get original symptom strings before encoding
        original_symptoms_in_col = df_orig[col].astype(str).str.strip().str.lower().str.replace(' ', '_').unique()
        symptoms_for_multiselect_set.update(s for s in original_symptoms_in_col if s not in ['nan', '', 'none', 'null', SYMPTOM_PLACEHOLDER])
available_symptoms_for_multiselect = sorted(list(symptoms_for_multiselect_set))

# Initialize weights dictionary for severity calculation
weights_dict = {}
if symptom_severity_df is not None:
    weights_dict = symptom_severity_df.set_index('symptom')['weight'].to_dict()

def calculate_df_severity(row, symptom_cols_list, current_weights_dict, s_mapping):
    """Calculates severity score for a row of symptoms (assumes symptoms are encoded)."""
    score = 0
    for col_name in symptom_cols_list:
        if col_name in row:
            symptom_key_encoded = row[col_name]
            # Reverse map encoded value to original string for weight lookup
            symptom_str = next((k for k, v in s_mapping.items() if v == symptom_key_encoded), SYMPTOM_PLACEHOLDER)
            if symptom_str != SYMPTOM_PLACEHOLDER:
                score += current_weights_dict.get(symptom_str, 0)
    return score

severity_q1_threshold = 0
severity_q2_threshold = 0

# Calculate severity score for the main DataFrame and determine thresholds
if symptom_severity_df is not None and weights_dict:
    df['severity_score'] = df.apply(lambda row: calculate_df_severity(row, symptom_string_cols, weights_dict, symptom_mapping), axis=1)
    if not df['severity_score'].empty and df['severity_score'].sum() > 0:
        if df['severity_score'].nunique() > 1:
            severity_q1_threshold = df['severity_score'].quantile(0.33)
            severity_q2_threshold = df['severity_score'].quantile(0.66)
            # Adjust thresholds if they are too close or identical
            if severity_q1_threshold == severity_q2_threshold:
                max_score_in_df = df['severity_score'].max()
                if max_score_in_df > 0:
                    severity_q1_threshold = max_score_in_df / 3
                    severity_q2_threshold = 2 * max_score_in_df / 3
                else: # All non-zero scores are identical
                    severity_q1_threshold = 1
                    severity_q2_threshold = 2
        elif df['severity_score'].nunique() == 1:
            unique_score = df['severity_score'].iloc[0]
            if unique_score > 0:
                severity_q1_threshold = unique_score / 3
                severity_q2_threshold = 2 * unique_score / 3
            else: # Single unique score is 0
                severity_q1_threshold = 1
                severity_q2_threshold = 2
    else: # Empty or all zero severity scores
        df['severity_score'] = 0
        st.warning("Kolom 'severity_score' kosong atau semua nol setelah kalkulasi. Menggunakan nilai default.")
        severity_q1_threshold = 1
        severity_q2_threshold = 2
else: # Symptom severity data not available
    df['severity_score'] = 0
    st.warning("Data keparahan gejala (Symptom-severity.csv) tidak tersedia atau kosong. 'severity_score' diatur ke 0.")
    severity_q1_threshold = 1
    severity_q2_threshold = 2

def categorize_severity(score, q1, q2):
    """Categorizes severity score into 'Rendah', 'Sedang', or 'Tinggi'."""
    if score <= q1:
        return "Rendah"
    elif score <= q2:
        return "Sedang"
    else:
        return "Tinggi"

# Ensure 'Disease' column exists before proceeding
if 'Disease' not in df.columns:
    st.error("Kolom target 'Disease' tidak ditemukan dalam dataset. Tidak dapat melanjutkan.")
    st.stop()

# --- KMeans Clustering for Training Data ---
# KMeans needs to be fit on the *same features* that will be used for prediction.
# Make sure 'cluster' is not in X_for_cluster initially, as it's the output.
cluster_features = [col for col in df.columns if col not in ['Disease', 'cluster']]
X_for_cluster = df[cluster_features].copy()

# Fit KMeans model (ensure n_init is set for newer scikit-learn versions)
try:
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_for_cluster)
    df['cluster'] = clusters
except Exception as e_kmeans:
    st.error(f"Gagal melakukan KMeans clustering: {e_kmeans}. Memastikan 'cluster' ditambahkan sebagai fitur dummy jika diperlukan.")
    df['cluster'] = 0 # Fallback if KMeans fails or if X_for_cluster is empty/invalid

# --- Prepare Data for Model Training and Prediction ---
X = df.drop('Disease', axis=1).copy()
y_series = df['Disease']
disease_encoder = LabelEncoder()
y = disease_encoder.fit_transform(y_series)
n_classes = len(disease_encoder.classes_)
if n_classes == 0:
    st.error("Tidak ada kelas penyakit yang terdeteksi setelah encoding. Periksa kolom 'Disease' Anda.")
    st.stop()

# Store expected feature order for consistency in user input for the model
expected_model_features = list(X.columns)

# --- Train-Test Split for Model Evaluation ---
if X.empty or len(y) == 0:
    st.warning("Tidak cukup data untuk train-test split dan evaluasi model.")
    X_train, X_test, y_train, y_test = pd.DataFrame(), pd.DataFrame(), np.array([]), np.array([])
else:
    try:
        # Ensure y is suitable for stratification (at least 2 samples per class)
        unique_classes, counts = np.unique(y, return_counts=True)
        if len(unique_classes) > 1 and all(c > 1 for c in counts):
            stratify_y = y
        else:
            stratify_y = None # Cannot stratify if any class has only one member

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=stratify_y)
    except ValueError as e_split:
        st.warning(f"Gagal melakukan stratified split ({e_split}). Melakukan split standar.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    except Exception as e_split_gen:
        st.warning(f"Gagal melakukan train-test split: {e_split_gen}. Melakukan split standar.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


## 🧪 Evaluasi Model Penyakit

if X_test.empty or len(y_test) == 0:
    st.info("Evaluasi model tidak dapat dilakukan karena tidak ada data test yang cukup.")
else:
    try:
        pred_raw = model.predict(X_test)
        y_pred = process_model_predictions(pred_raw)

        if not isinstance(y_pred, np.ndarray) or y_pred.ndim != 1 or (len(y_test) > 0 and len(y_pred) != len(y_test)):
            st.error(f"Output y_pred tidak valid. Shape: {getattr(y_pred, 'shape', 'N/A')}. Gagal menghitung metrik.")
            y_pred = np.array([-1] * len(y_test)) if len(y_test) > 0 else np.array([]) # Fallback for metrics

        prob_raw = model.predict_proba(X_test)
        y_prob = process_model_probabilities(prob_raw, n_classes, len(X_test))

        accuracy = 0.0
        precision = 0.0
        recall = 0.0

        if len(y_pred) == len(y_test) and len(y_test) > 0 and n_classes > 0:
            accuracy = accuracy_score(y_test, y_pred)
            # Use 'labels' parameter with unique labels present in y_true or y_pred for robust metric calculation
            unique_y_test_pred = np.unique(np.concatenate((y_test, y_pred)))
            precision = precision_score(y_test, y_pred, average='macro', zero_division=0, labels=unique_y_test_pred)
            recall = recall_score(y_test, y_pred, average='macro', zero_division=0, labels=unique_y_test_pred)

        cm = confusion_matrix(y_test, y_pred, labels=range(n_classes)) if len(y_pred) == len(y_test) and len(y_test) > 0 else np.array([[0]*n_classes]*n_classes)

        auc_score = 0.0
        can_calc_auc = isinstance(y_prob, np.ndarray) and y_prob.ndim == 2 and y_prob.shape[0] == len(y_test) and y_prob.shape[1] == n_classes and len(np.unique(y_test)) > 1

        if can_calc_auc:
            try:
                y_bin = label_binarize(y_test, classes=range(n_classes))
                if n_classes == 2:
                    auc_score = roc_auc_score(y_test, y_prob[:, 1])
                elif y_bin.shape[1] == n_classes:
                    auc_score = roc_auc_score(y_bin, y_prob, multi_class='ovr', average='macro')
            except ValueError:
                st.warning("ROC AUC calculation failed due to ValueError (e.g., single class present in test set). Skipping ROC AUC.")
            except Exception as e_auc:
                st.warning(f"ROC AUC calculation failed: {e_auc}. Skipping ROC AUC.")

        # --- Display Metrics ---
        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            st.metric("Classes", f"{n_classes}")
        with row1_col2:
            st.metric("Test Samples", f"{len(y_test)}")

        row2_col1, row2_col2, row2_col3, row2_col4 = st.columns(4)
        with row2_col1:
            st.metric("Accuracy", f"{accuracy:.2f}")
        with row2_col2:
            st.metric("Precision", f"{precision:.2f}")
        with row2_col3:
            st.metric("Recall", f"{recall:.2f}")
        with row2_col4:
            st.metric("ROC AUC", f"{auc_score:.2f}" if auc_score > 0 else "N/A")

        # --- Visualization Options ---
        viz_option = st.selectbox("Pilih visualisasi:", ["Tidak Ada", "ROC Curve", "Confusion Matrix"])

        target_names_str = [str(cls_name) for cls_name in disease_encoder.classes_]

        if viz_option == "ROC Curve" and can_calc_auc and auc_score > 0:
            st.subheader("Kurva ROC (One-vs-Rest)")
            fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
            fig_roc.set_facecolor(DARK_BG_COLOR)
            ax_roc.set_facecolor(DARK_BG_COLOR)

            for spine in ax_roc.spines.values():
                spine.set_edgecolor(OUTLINE_COLOR)

            ax_roc.tick_params(colors=LIGHT_TEXT_COLOR, which='both')

            y_bin_roc = label_binarize(y_test, classes=range(n_classes))

            if n_classes == 2:
                fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
                ax_roc.plot(fpr, tpr, label=f'{target_names_str[1]} (AUC = {auc_score:.2f})', linewidth=2)
            elif y_bin_roc.shape[1] == n_classes:
                for i in range(n_classes):
                    if i < len(target_names_str):
                        try:
                            if i in np.unique(y_test): # Plot only if class is present in y_test
                                fpr, tpr, _ = roc_curve(y_bin_roc[:, i], y_prob[:, i])
                                auc_val_class = roc_auc_score(y_bin_roc[:, i], y_prob[:, i])
                                ax_roc.plot(fpr, tpr, label=f'{target_names_str[i]} (AUC = {auc_val_class:.2f})', linewidth=2)
                        except ValueError:
                            pass # Skip if ROC cannot be calculated for this class

            ax_roc.plot([0, 1], [0, 1], linestyle='--', color=OUTLINE_COLOR, linewidth=1.5)

            legend_fontsize = 'x-small' if n_classes > 5 else None
            legend_roc = ax_roc.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=legend_fontsize,
                                       facecolor=DARK_BG_COLOR, edgecolor=OUTLINE_COLOR)
            for text in legend_roc.get_texts():
                text.set_color(LIGHT_TEXT_COLOR)
            if legend_roc.get_title():
                legend_roc.get_title().set_color(LIGHT_TEXT_COLOR)

            ax_roc.set_title('Kurva ROC (One-vs-Rest)', fontsize=15, color=LIGHT_TEXT_COLOR)
            ax_roc.set_xlabel('False Positive Rate', fontsize=12, color=LIGHT_TEXT_COLOR)
            ax_roc.set_ylabel('True Positive Rate', fontsize=12, color=LIGHT_TEXT_COLOR)
            ax_roc.grid(True, alpha=0.3, color=GRID_COLOR)
            st.pyplot(fig_roc)
            plt.close(fig_roc)

        elif viz_option == "Confusion Matrix":
            st.subheader("Confusion Matrix")
            fig_cm, ax_cm = plt.subplots(figsize=(max(8, n_classes*0.7), max(6, n_classes*0.6)))
            fig_cm.set_facecolor(DARK_BG_COLOR)

            sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', ax=ax_cm,
                        xticklabels=target_names_str, yticklabels=target_names_str,
                        linewidths=.5, linecolor=HEATMAP_LINE_COLOR,
                        annot_kws={"size": 10 if n_classes < 10 else 8, "color": "black"})

            for spine in ax_cm.spines.values():
                spine.set_edgecolor(OUTLINE_COLOR)

            ax_cm.set_title('Confusion Matrix', fontsize=15, color=LIGHT_TEXT_COLOR)
            ax_cm.set_xlabel('Prediksi Label', fontsize=12, color=LIGHT_TEXT_COLOR)
            ax_cm.set_ylabel('Label Sebenarnya', fontsize=12, color=LIGHT_TEXT_COLOR)

            ax_cm.tick_params(axis='x', colors=LIGHT_TEXT_COLOR)
            ax_cm.tick_params(axis='y', colors=LIGHT_TEXT_COLOR)
            plt.xticks(rotation=45, ha='right', fontsize=10, color=LIGHT_TEXT_COLOR)
            plt.yticks(rotation=0, fontsize=10, color=LIGHT_TEXT_COLOR)

            if hasattr(ax_cm.collections[0], 'colorbar') and ax_cm.collections[0].colorbar is not None:
                cbar = ax_cm.collections[0].colorbar
                cbar.ax.tick_params(colors=LIGHT_TEXT_COLOR)

            plt.tight_layout(pad=1.5)
            st.pyplot(fig_cm)
            plt.close(fig_cm)

    except Exception as e_eval:
        st.error(f"Gagal melakukan evaluasi model: {e_eval}")
        st.error(traceback.format_exc())


## 🔍 Prediksi Penyakit

with st.form("prediction_form"):
    selected_symptoms_raw = st.multiselect(
        "Pilih gejala yang dialami:",
        options=available_symptoms_for_multiselect,
        help="Pilih satu atau lebih gejala dari daftar"
    )
    predict_btn = st.form_submit_button("🧠 Prediksi Penyakit", use_container_width=True)

if predict_btn:
    if selected_symptoms_raw:
        try:
            # 1. Initialize user input data with placeholder for all symptom columns
            # This ensures all symptom columns expected by the model are present and
            # initially set to represent an 'absent' symptom.
            user_input_data = {}
            for feature_col in expected_model_features:
                if 'Symptom_' in feature_col:
                    user_input_data[feature_col] = symptom_mapping.get(SYMPTOM_PLACEHOLDER, 0)
                elif feature_col == 'severity_score' or feature_col == 'cluster':
                    user_input_data[feature_col] = 0 # Default numerical value
                else:
                    user_input_data[feature_col] = 0 # Fallback for any other unexpected numerical features

            # 2. Fill the symptom columns based on user's selected symptoms by finding available slots.
            # This is the critical change to ensure order-independence.
            filled_symptom_cols_count = 0
            for symptom_str in selected_symptoms_raw:
                # Find an available Symptom_X column (one that currently holds the placeholder value)
                found_slot = False
                for col_name in symptom_string_cols:
                    if user_input_data.get(col_name) == symptom_mapping.get(SYMPTOM_PLACEHOLDER, 0):
                        user_input_data[col_name] = symptom_mapping.get(symptom_str, symptom_mapping[SYMPTOM_PLACEHOLDER])
                        found_slot = True
                        filled_symptom_cols_count += 1
                        break # Move to the next selected symptom
                if not found_slot:
                    st.warning(f"Semua kolom gejala telah diisi. Gejala '{symptom_str}' mungkin diabaikan karena tidak ada slot kosong. Jumlah gejala yang dipilih melebihi yang dapat ditampung model.")
                    break # Move to the next selected symptom
                if not found_slot:
                    st.warning(f"Semua kolom gejala telah diisi. Gejala '{symptom_str}' mungkin diabaikan karena tidak ada slot kosong. Jumlah gejala yang dipilih melebihi yang dapat ditampung model.")
                    break # Stop if all symptom columns are filled

            # 3. Calculate severity_score based on the original string symptoms selected by the user
            current_user_severity_score = 0
            if symptom_severity_df is not None and weights_dict:
                for symptom_name_str in selected_symptoms_raw:
                    current_user_severity_score += weights_dict.get(symptom_name_str, 0)
            user_input_data['severity_score'] = current_user_severity_score

            # 4. Create the DataFrame for prediction, ensuring all expected_model_features are present
            user_df = pd.DataFrame([user_input_data], columns=expected_model_features)

            # 5. Calculate user cluster (if 'cluster' is an expected feature by the model)
            if 'cluster' in expected_model_features:
                features_for_user_cluster = user_df.drop(columns=['cluster'], errors='ignore')
                if not X_for_cluster.empty and kmeans: # Ensure KMeans was fitted and not empty
                    try:
                        user_cluster = kmeans.predict(features_for_user_cluster)
                        user_df['cluster'] = user_cluster[0]
                    except Exception as e_kmeans_predict:
                        st.warning(f"Gagal memprediksi cluster untuk input pengguna: {e_kmeans_predict}. Menggunakan cluster default (0).")
                        user_df['cluster'] = 0
                else:
                    user_df['cluster'] = 0 # Default if KMeans wasn't properly fitted or X_for_cluster is empty

            # 6. Make prediction using the model
            prob_raw_user = model.predict_proba(user_df)
            probabilities = process_model_probabilities(prob_raw_user, n_classes, 1)

            if probabilities is not None and probabilities.ndim == 2 and probabilities.shape[1] == n_classes:
                predicted_idx = np.argmax(probabilities)
                confidence = probabilities[0, predicted_idx]
                disease_name = disease_encoder.inverse_transform([predicted_idx])[0]

                st.success("### 📋 Hasil Prediksi")
                col1_res, col2_res = st.columns([2,1])
                with col1_res: st.markdown(f"**Penyakit:** <span style='font-size: 1.2em; color: #28a745;'>**{str(disease_name).strip()}**</span>", unsafe_allow_html=True)
                with col2_res: st.markdown(f"**Keyakinan:** <span style='font-size: 1.2em;'>{confidence:.1%}</span>", unsafe_allow_html=True)
                st.markdown("---")

                if desc_df is not None:
                    desc_row = desc_df[desc_df['Disease'].str.strip().str.lower() == str(disease_name).strip().lower()]
                    if not desc_row.empty: st.info(f"**📖 Deskripsi:** {desc_row['Description'].values[0]}")
                    else: st.caption(f"Deskripsi tidak tersedia untuk '{str(disease_name).strip()}'.")

                if precaution_df is not None:
                    precaution_row = precaution_df[precaution_df['Disease'].str.strip().str.lower() == str(disease_name).strip().lower()]
                    if not precaution_row.empty:
                        precautions = [str(p).strip() for p in precaution_row.iloc[0, 1:].values if pd.notna(p) and str(p).strip().lower() not in ['nan', '']]
                        if precautions:
                            st.warning("**🛡️ Tindakan Pencegahan & Saran:**")
                            for i, p_text in enumerate(precautions, 1): st.write(f"{i}. {p_text}")
                        else: st.caption(f"Tindakan pencegahan tidak spesifik atau tidak tersedia untuk '{str(disease_name).strip()}'.")
                    else: st.caption(f"Informasi tindakan pencegahan tidak ditemukan untuk '{str(disease_name).strip()}'.")

                st.markdown("---")

                if probabilities is not None and probabilities.shape[1] > 1:
                    st.markdown("### 📊 Top 5 Kemungkinan Diagnosis")
                    top_indices = np.argsort(probabilities[0])[::-1][:min(5, n_classes)]
                    top_diseases = disease_encoder.inverse_transform(top_indices)
                    top_probs = probabilities[0, top_indices]

                    results_data = {'Penyakit': [str(d).strip() for d in top_diseases], 'Probabilitas': top_probs}
                    results_df = pd.DataFrame(results_data)

                    chart_data = results_df.set_index('Penyakit')
                    st.bar_chart(chart_data, height=300)
                    st.dataframe(results_df.style.format({'Probabilitas': '{:.1%}'}), use_container_width=True, hide_index=True)

                if symptom_severity_df is not None and weights_dict:
                    severity_category = categorize_severity(current_user_severity_score, severity_q1_threshold, severity_q2_threshold)
                    st.info(f"ℹ️ **Total Skor Keparahan Gejala (berdasarkan input):** {current_user_severity_score} (Kategori: **{severity_category}**).\n\nAmbang batas kategori: Rendah ≤ {severity_q1_threshold:.1f}, Sedang ≤ {severity_q2_threshold:.1f}, Tinggi > {severity_q2_threshold:.1f}. Skor ini adalah salah satu dari berbagai faktor yang dipertimbangkan model.")
                else:
                    st.info("ℹ️ Skor keparahan gejala tidak dapat dihitung (data 'Symptom-severity.csv' tidak tersedia atau kosong), sehingga kategori tidak dapat ditentukan.")

            else:
                st.error("Tidak dapat menghitung probabilitas prediksi. Model mungkin tidak mengembalikan output yang diharapkan atau jumlah kelas tidak sesuai.")

        except KeyError as e_key:
            st.error(f"Terjadi kesalahan pemetaan fitur (KeyError): '{e_key}'. Ini mungkin bug internal atau ketidaksesuaian data/kolom.")
            st.error(traceback.format_exc())
        except ValueError as e_val:
            st.error(f"Terjadi kesalahan dalam pemrosesan input atau prediksi (ValueError): {e_val}.")
            st.error(traceback.format_exc())
        except Exception as e_pred:
            st.error(f"Terjadi kesalahan umum saat memproses prediksi: {e_pred}")
            st.error(traceback.format_exc())

    else:
        st.warning("⚠️ Silakan pilih minimal satu gejala untuk melakukan prediksi.")

st.markdown("---")
st.caption("⚕️ *Aplikasi ini untuk tujuan informasi dan edukasi saja. Selalu konsultasikan dengan profesional medis atau dokter untuk diagnosis dan perawatan yang akurat.*")
