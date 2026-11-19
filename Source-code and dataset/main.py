import os, logging, warnings, gc

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf

tf.get_logger().setLevel(logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K

import numpy as np
import matplotlib

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib.pyplot as plt

logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

from shap_explainability import SHAPExplainer
from lime_explainability import LIMEExplainer
from reporting import export_plots_to_html
from data_preprocessing import TelcoDataProcessor
from hybrid_neural_network_model import HybridModel
from model_evaluation import (plot_class_distribution, detailed_metrics, plot_evaluation, validate_anchor_rule,
                              plot_preprocessed_features, plot_anchor_rules_table)
from models_comparison import compare_models
from customer_form import CustomerSelectionForm

#==================================Method calls inside Main========================================================
def get_callbacks(patience=3):
    return [EarlyStopping(monitor='val_f1', mode='max', patience=patience, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_f1', mode='max', factor=0.5, patience=patience, min_lr=1e-6, verbose=0)]

def to_3d(X):
    return X.reshape(X.shape[0], X.shape[1], 1)

def prep_leakage_safe_data(x_train_raw, y_train, x_test_raw, val_size, processor, numerical_cols):
    # Handles split, fold-safe preprocessing, scaling, and SMOTE for baseline & deployment models.
    # 1. Split
    x_tr_raw, x_va_raw, y_tr, y_va = train_test_split(x_train_raw, y_train, test_size=val_size, random_state=42,
                                                      stratify=y_train)
    # 2. Preprocess (Fit on train, transform on val/test)
    state = processor.make_preprocessing_state()
    x_tr_df, state = processor.fit_transform_preprocessing(x_tr_raw, state)
    x_va_df = processor.transform_preprocessing(x_va_raw, state)
    x_te_df = processor.transform_preprocessing(x_test_raw, state)

    # 3. Convert to numpy arrays
    x_tr = x_tr_df.values.astype(np.float32)
    x_va = x_va_df.values.astype(np.float32)
    x_te = x_te_df.values.astype(np.float32)

    # 4. Scale
    num_idx = [state["feature_names"].index(c) for c in numerical_cols if c in state["feature_names"]]
    scaler = StandardScaler()

    x_tr[:, num_idx] = scaler.fit_transform(x_tr[:, num_idx])
    x_va[:, num_idx] = scaler.transform(x_va[:, num_idx])
    x_te[:, num_idx] = scaler.transform(x_te[:, num_idx])
    # 5. Apply SMOTE ( on training data only)
    x_tr_res, y_tr_res, _ = processor.apply_smote(x_tr, y_tr)

    return x_tr_res, y_tr_res, x_va, y_va, x_te, state["feature_names"]


# ======================== Main Project Execution ==============================================================
def main(data_path='WA_Fn-UseC_-Telco-Customer-Churn.csv', random_state=42):

    all_figures = []
    # STEP 2: Data Loading & Row-wise Cleaning
    print("\n=== 2. Data Loading & Row-wise Cleaning ===")
    processor = TelcoDataProcessor(data_path)
    df_cleaned = processor.clean_data()
    y = df_cleaned['Churn'].values.astype(np.int32)
    X_df = df_cleaned.drop('Churn', axis=1)
    indices = df_cleaned.index.values

    # Train (80%) and Test (20%) splitting
    x_train_df_raw, x_test_df_raw, y_train_full, y_test, _, indices_test = \
    train_test_split(X_df, y, indices, test_size=0.2, random_state=random_state, stratify=y)
    x_train_df_raw.to_csv('train_split.csv', index=False)
    x_test_df_raw.to_csv('test_split.csv', index=False)
    print(f"Raw training pool: {x_train_df_raw.shape}, Raw test set: {x_test_df_raw.shape}")

    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AverageMonthlyCharge']

    # STEP 3: Class Distribution Visualisation
    print("\n=== 3. Class Distribution Visualisation ===")
    non_churn, churn = int((y_train_full == 0).sum()), int((y_train_full == 1).sum())
    before_sampling = {"Non-Churn": non_churn, "Churn": churn}
    after_sampling = {"Non-Churn": max(non_churn, churn), "Churn": max(non_churn, churn)}
    all_figures.append(plot_class_distribution(before_sampling, after_sampling))

    # STEP 4: Stratified 5-Fold Cross-Validation
    print("\n=== 4. Stratified 5-Fold Cross-Validation Training ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    oof_pred, test_pred_folds = np.zeros(len(x_train_df_raw)), np.zeros(len(x_test_df_raw))

    for fold, (train_idx, val_idx) in enumerate(skf.split(x_train_df_raw, y_train_full)):
        print(f"\n--- Training Fold {fold + 1} / 5 ---")
        x_tr_raw, x_va_raw = x_train_df_raw.iloc[train_idx].copy(), x_train_df_raw.iloc[val_idx].copy()
        y_tr, y_va = y_train_full[train_idx], y_train_full[val_idx]

        fold_state = processor.make_preprocessing_state()
        x_tr_df, fold_state = processor.fit_transform_preprocessing(x_tr_raw, fold_state)
        x_va_df = processor.transform_preprocessing(x_va_raw, fold_state)

        num_idx = [fold_state["feature_names"].index(c) for c in numerical_cols if c in fold_state["feature_names"]]
        x_tr, x_va = x_tr_df.values.astype(np.float32), x_va_df.values.astype(np.float32)

        scaler = StandardScaler()
        x_tr[:, num_idx], x_va[:, num_idx] = scaler.fit_transform(x_tr[:, num_idx]), scaler.transform(x_va[:, num_idx])
        x_tr_res, y_tr_res, _ = processor.apply_smote(x_tr, y_tr)

        model = HybridModel.build(input_shape=(x_tr_res.shape[1], 1), lr=0.0003)
        model.fit(x_tr_res, y_tr_res, epochs=30, batch_size=64, validation_data=(to_3d(x_va), y_va),
                  callbacks=get_callbacks(), verbose=1)

        oof_pred[val_idx] = model.predict(to_3d(x_va), verbose=0).ravel()

        x_te = processor.transform_preprocessing(x_test_df_raw, fold_state).values.astype(np.float32)
        x_te[:, num_idx] = scaler.transform(x_te[:, num_idx])
        test_pred_folds += model.predict(to_3d(x_te), verbose=0).ravel() / 5

    # STEP 5: Threshold tuning on OOF
    print("\n=== 5. Finding Optimal Threshold on OOF Predictions ===")
    thresh_scores = [(t, f1_score(y_train_full, (oof_pred >= t).astype(int))) for t in np.arange(0.1, 0.9, 0.01)]
    best_thresh, best_f1 = max(thresh_scores, key=lambda x: x[1])
    print(f"Optimal threshold: {best_thresh:.2f} (F1: {best_f1:.4f})")

    # STEP 6: Evaluation on Test Set
    print("\n=== 6. Evaluating Ensemble Model on Untouched 20% Test Set ===")
    y_pred = (test_pred_folds >= best_thresh).astype(int)
    all_figures.append(plot_evaluation(y_test, test_pred_folds, y_pred))
    metrics = detailed_metrics(y_test, y_pred, test_pred_folds)
    hybrid_metrics = {**{k: v * 100 for k, v in metrics.items() if k in ['accuracy', 'precision', 'recall', 'f1']},
                      "roc_auc": metrics["roc_auc"], "y_pred": y_pred, "y_proba": test_pred_folds}

    # STEP 7: Leakage-Safe Deployment Model
    print("\n=== 7. Leakage-Safe Deployment Model ===")
    dep_x_tr_res, dep_y_tr_res, dep_x_va, dep_y_va, dep_x_te, feature_names = \
        prep_leakage_safe_data(x_train_df_raw, y_train_full, x_test_df_raw, 0.10, processor, numerical_cols)

    deploy_model = HybridModel.build(input_shape=(dep_x_tr_res.shape[1], 1), lr=0.0003)
    deploy_model.fit(dep_x_tr_res, dep_y_tr_res, epochs=30, batch_size=64, validation_data=(to_3d(dep_x_va), dep_y_va),
                     callbacks=get_callbacks(5), verbose=1)

    inp_2d = tf.keras.Input(shape=(dep_x_tr_res.shape[1],))
    shap_lime_model = tf.keras.Model(inputs=inp_2d,
                                     outputs=deploy_model(tf.keras.layers.Reshape((dep_x_tr_res.shape[1], 1))(inp_2d)))

    print(f"\nFinal list of {len(feature_names)} preprocessed features:")
    for i, feat in enumerate(feature_names, 1):
        print(f"{i}. {feat}")
    print("\n=== Visualizing Final Preprocessed Features ===")
    all_figures.insert(1, plot_preprocessed_features(feature_names))

    # STEP 8: Leakage-Safe Baseline Comparison
    print("\n=== 8. Leakage-Safe Baseline Comparison ===")
    base_x_tr_res, base_y_tr_res, base_x_va, base_y_va, base_x_te, _ = \
        prep_leakage_safe_data(x_train_df_raw, y_train_full, x_test_df_raw, 0.20, processor, numerical_cols)

    cnn_fig, bilstm_fig, comp_fig = compare_models(base_x_tr_res, base_y_tr_res, base_x_va, base_y_va, base_x_te,
                                                   y_test, hybrid_metrics=hybrid_metrics)
    all_figures.extend([cnn_fig, bilstm_fig, comp_fig])
    gc.collect()

    # STEP 9: Customer Selection GUI
    print("\n=== 9. Customer Selection GUI Form ===")
    form = CustomerSelectionForm(processor, indices_test, len(indices_test) - 1)
    sample_idx, raw_customer = form.get_selected_index()

    if sample_idx is None:
        print("No customer selected. Exiting.")
        export_plots_to_html(all_figures)
        K.clear_session(); gc.collect()
        return

    x_train_2d, x_test_2d = dep_x_tr_res.reshape(dep_x_tr_res.shape[0], -1), dep_x_te
    # Original selected customer instance being explained (preprocessed test data)
    # Will be used in shap and lime anchor rules argument of function validate_anchor_rule()
    x_original = x_test_2d[sample_idx]

    assert x_train_2d.shape[1] == x_test_2d.shape[1] == feature_names.__len__()
    CUST_ACTUAL_ID = processor.get_customer_original_data(sample_idx, indices_test).get("customerID", "Unknown")

    # STEP 10-13: Explainability
    try:
        print("\n=== 10. SHAP Global Explainability ===")
        shap_explainer = SHAPExplainer(shap_lime_model, feature_names, x_train_2d, x_test_2d,
                                       background_samples=50, algorithm="gradient")
        shap_explainer.plot_summary()
        shap_top_features = shap_explainer.get_top_shap_features()
        print("Top SHAP Features:", shap_top_features)

        if raw_customer: all_figures.append(CustomerSelectionForm.customer_profile_figure(raw_customer))

        print("\n=== 11. SHAP Local Explainability ===")
        all_figures.append(shap_explainer.plot_waterfall(cust_actual_id=CUST_ACTUAL_ID, sample_idx=sample_idx))
        all_figures.append(shap_explainer.shap_force_plot(cust_actual_id=CUST_ACTUAL_ID, sample_idx=sample_idx))
        all_figures.append(shap_explainer.shap_decision_plot(cust_actual_id=CUST_ACTUAL_ID, sample_idx=sample_idx))

        # --- Validate SHAP Anchor Rule  ---
        shap_rule_conditions = [{'feature': 'tenure', 'op': '<=', 'threshold': -1.24},
                                {'feature': 'MonthlyCharges', 'op': '>=', 'threshold': 0.84},
                                {'feature': 'Contract_Month-to-month', 'op': '==', 'threshold': 1.0}]
        shap_cov, shap_prec = validate_anchor_rule("SHAP Rule (A_SHAP)", shap_rule_conditions, shap_lime_model,
                                                  x_test_2d, feature_names, x_original)

        print("\n=== 12. LIME Local Explainability ===")
        lime_explainer = LIMEExplainer(shap_lime_model, feature_names, x_train_2d, x_test_2d)
        all_figures.append(lime_explainer.lime_bar_chart_plot(cust_actual_id=CUST_ACTUAL_ID, sample_idx=sample_idx, num_features=10))
        all_figures.append(lime_explainer.lime_decision_path_plot(cust_actual_id=CUST_ACTUAL_ID, sample_idx=sample_idx))
        all_figures.append(lime_explainer.lime_heatmap_plot(cust_actual_id=CUST_ACTUAL_ID, sample_idx=sample_idx, num_features=10))

        # --- Validate LIME Anchor Rule  ---
        lime_rule_conditions = [{'feature': 'tenure', 'op': '<=', 'threshold': -0.96},
                                {'feature': 'Contract_Month-to-month', 'op': '==', 'threshold': 1.0},
                                {'feature': 'MonthlyCharges', 'op': '>', 'threshold': 0.83}]
        lime_cov, lime_prec = validate_anchor_rule("LIME Rule (A_LIME)", lime_rule_conditions, shap_lime_model,
                                                   x_test_2d, feature_names, x_original)
        # --- Add Anchor Rules Table ---
        print("\n=== 13. Generating Anchor Rules Table ===")
        all_figures.append(plot_anchor_rules_table((shap_cov, shap_prec), (lime_cov, lime_prec)))

    except Exception as e:
        print("Explainability error:", str(e))
        plt.show()

    # STEP 14: Export report results
    export_plots_to_html(all_figures)
    del all_figures; K.clear_session(); gc.collect()

if __name__ == "__main__":
    main()