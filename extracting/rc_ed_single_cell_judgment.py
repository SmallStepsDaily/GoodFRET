"""
单细胞FRET变化判断函数
1. 使用control组拟合函数
2. 使用加药组组拟合函数
3. 单细胞尺度利用区域效率查看符合哪个拟合函数判断是否发生了 FRET 变化
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 支持多组模型的表达式输入形式（如 {'control': '0.8*x + 0.1', 'AFA': '0.6*x + 0.2'}）
def parse_model(expr):
    def func(x):
        return eval(expr, {"x": x, "np": np})
    return func

def analyze_fret_data(input_file, model_dict, treatment_col='Metadata_treatment',
                      output_path=None, save_plot=True, plot_file=None):
    df = pd.read_csv(input_file)
    print(f"读取文件成功: {input_file}，共 {len(df)} 条记录")

    if 'Ed_region_mean' not in df.columns or 'Rc_region_mean' not in df.columns:
        raise ValueError("缺少 Ed_region_mean 或 Rc_region_mean 列")

    ed = df['Ed_region_mean'].values
    rc = df['Rc_region_mean'].values
    treatments = df[treatment_col].astype(str).values

    parsed_models = {k: parse_model(v) for k, v in model_dict.items()}

    judge_result = []
    nearest_pred = []
    ed_diff = []

    for i in range(len(df)):
        trt = treatments[i]
        rc_val = rc[i]
        ed_val = ed[i]
        if np.isnan(ed_val):
            judge_result.append(np.nan)
            nearest_pred.append(np.nan)
            ed_diff.append(np.nan)
            continue

        if trt not in parsed_models or 'control' not in parsed_models:
            judge_result.append("")
            nearest_pred.append(np.nan)
            ed_diff.append(np.nan)
            continue

        control_model = parsed_models['control']
        drug_model = parsed_models[trt]

        control_pred = control_model(rc_val)
        drug_pred = drug_model(rc_val)

        control_error = abs(ed_val - control_pred)
        drug_error = abs(ed_val - drug_pred)

        is_drug = drug_error < control_error
        if np.isnan(drug_pred):
            judge_result.append("")
        else:
            judge_result.append(trt if is_drug else 'control')
        nearest_pred.append(drug_pred if is_drug else control_pred)
        ed_diff.append(control_pred - (drug_pred if is_drug else control_pred))

    df['FRET_Judge'] = judge_result
    df['Near_Ed'] = nearest_pred
    df['Ed_diff'] = ed_diff

    output_file = os.path.join(str(output_path), "Rc-Ed_FRET_analyzed.csv")
    df.to_csv(output_file, index=False)
    print(f"分析结果已保存至: {output_path}")

    if save_plot:
        plot_fret_analysis(df, parsed_models, output_path)

def plot_fret_analysis(df, model_funcs, output_path):
    plt.figure(figsize=(10, 6))
    df = df.dropna()
    rc_values = df['Rc_region_mean'].values
    ed_values = df['Ed_region_mean'].values
    judge_results = df['FRET_Judge'].values

    for label in sorted(set(judge_results)):
        mask = judge_results == label
        if label == "":
            continue
        elif label == 'control':
            plt.scatter(rc_values[mask], ed_values[mask], c='blue', s=30, alpha=0.6, label='Near control')
        else:
            plt.scatter(rc_values[mask], ed_values[mask], label=f'Near {label}', s=30, alpha=0.6)

    rc_range = np.linspace(0, max(rc_values), 500)

    if 'control' in model_funcs:
        plt.plot(rc_range, model_funcs['control'](rc_range), 'b-', label='control_ed', linewidth=2)
    for label in set(judge_results):
        if label != 'control' and label != 'unknown' and label in model_funcs:
            plt.plot(rc_range, model_funcs[label](rc_range), '--', linewidth=2, label=f'{label}_ed')

    plt.xlabel('Rc')
    plt.ylabel('Ed')
    plt.title('FRET Data Analysis (Non-control Points)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    plt.tight_layout()


    output_file = os.path.join(output_path, 'Rc-Ed_drug_classification.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"图像已保存至: {output_file}")
    plt.close()

if __name__ == "__main__":
    folder_path = r'C:\Code\python\csv_data\gl\EGFR-GRB2实验数据\A549\20250514'
    input_csv = f"{folder_path}\FRET.csv"

    # EGFR 拟合靶点
    EGFR_model_exprs = {
        'control': '0.4449475861028414 * x / (1.0 + x)',
        'DAC': '0.1122966659895491 * x',
        'ALM': '0.09516652556445405 * x ',
        'GEF': '0.09310420785005083 * x',
        'AFA': '0.08681578444875934 * x',
        'OSI': '0.05591866357178568 * x',
        'VIN': '0.4109161217383655 * x / (1.0 + x)',
    }

    analyze_fret_data(input_csv, EGFR_model_exprs, output_path=folder_path, save_plot=True)
