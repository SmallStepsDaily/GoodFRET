# lda_train.py
import os
import sys
import pandas as pd
import numpy as np
import joblib
from io import BytesIO
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

def compute_and_save_model(csv_path, ptype):
    df = pd.read_csv(csv_path)

    # 元数据列
    meta_cols = ['Metadata_treatment', 'Metadata_cell', 'Metadata_hour', 'Metadata_concentration']
    assert all(col in df.columns for col in meta_cols), "Missing metadata columns in input CSV."
    feature_cols = [col for col in df.columns if
                        not col.startswith(
                            'Metadata_') and col != 'ObjectNumber' and col != 'Label' and col != 'ImageNumber']
    drug_name = df[df['Metadata_treatment'] != 'control']['Metadata_treatment'].unique()[0]
    control = df[df['Metadata_treatment'] == 'control']
    drug = df[df['Metadata_treatment'] == drug_name]

    combined_data = pd.concat([drug, control])

    X = combined_data[feature_cols]
    y = combined_data['Metadata_treatment']

    metadata = combined_data[['ObjectNumber', 'Metadata_cell' ,'Metadata_site', 'Metadata_concentration', 'Metadata_hour']].copy()

    # 下采样
    if len(drug) < len(control) + 20:
        from imblearn.under_sampling import RandomUnderSampler
        rus = RandomUnderSampler(random_state=42)
        X_bal, y_bal = rus.fit_resample(X, y)
    else:
        X_bal, y_bal = X, y

    X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.3, random_state=42, stratify=y_bal)
    is_test = pd.Series(False, index=X.index)
    is_test[X_test.index] = True

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lda = LinearDiscriminantAnalysis()
    model = CalibratedClassifierCV(lda, cv=5)
    model.fit(X_train_scaled, y_train)

    print("Validation Classification Report:")
    print(classification_report(y_test, model.predict(X_test_scaled)))

    lb = LabelBinarizer()
    lb.fit(y_train)
    drug_label = list(lb.classes_).index(drug_name)

    X_scaled = scaler.transform(X)
    all_probs = model.predict_proba(X_scaled)[:, drug_label]

    all_results = pd.DataFrame({
        'Metadata_treatment': y,
        'Predicted_Probability': all_probs,
    })
    all_results = pd.concat([metadata.reset_index(drop=True), all_results.reset_index(drop=True)], axis=1)

    # 画图
    y_train_bin = lb.transform(y_train)
    y_test_bin = lb.transform(y_test)
    y_train_prob = model.predict_proba(X_train_scaled)[:, drug_label]
    y_test_prob = model.predict_proba(X_test_scaled)[:, drug_label]
    fpr_train, tpr_train, _ = roc_curve(y_train_bin.ravel(), y_train_prob, pos_label=drug_label)
    fpr_test, tpr_test, _ = roc_curve(y_test_bin.ravel(), y_test_prob, pos_label=drug_label)
    auc_train = auc(fpr_train, tpr_train)
    auc_test = auc(fpr_test, tpr_test)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    ax1.plot(fpr_train, tpr_train, color='blue', lw=2, label=f'Training ROC (AUC = {auc_train:.2f})')
    ax1.plot(fpr_test, tpr_test, color='darkorange', lw=2, label=f'Test ROC (AUC = {auc_test:.2f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_title('ROC Curve')
    ax1.legend(loc='lower right')

    sns.histplot(data=all_results, x='Predicted_Probability', hue='Metadata_treatment', bins=20, kde=True, ax=ax2)
    ax2.set_title('Predicted Probability Distribution')

    sns.boxplot(x='Metadata_treatment', y='Predicted_Probability', data=all_results, ax=ax3)
    ax3.set_title('Boxplot by Treatment')

    plt.tight_layout()

    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    fig_path = os.path.join("model", f"{base_name}_LDA_Train_Result.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"保存图像至: {fig_path}")

    # 保存模型
    cell = df['Metadata_cell'].iloc[0]
    model_name = f"{ptype}-{cell}-{drug_name}.pkl"
    os.makedirs("model", exist_ok=True)
    model_path = os.path.join("model", model_name)
    joblib.dump((model, scaler), model_path)
    print(f"保存模型至: {model_path}")

if __name__ == '__main__':
    compute_and_save_model(r"C:\Code\python\csv_data\gl\EGFR-GRB2实验数据\A549\模型数据\FB_BF_DAC.csv", 'BF')
