import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score
import os

'''
Rc-Region_Ed拟合算法
- 使用非线性拟合模型（edmax * Rc / (kd + Rc)）
- 同时进行线性拟合（a * Rc）
- 加入Region_pixels_sum作为权重
- Control组同时展示线性与非线性拟合
'''

# ================== 模型定义 ==================
def ed_nonlinear(rc, edmax, kd):
    return edmax * rc / (kd + rc)

def ed_linear(rc, a):
    return a * rc

def get_initial_params(x_data, y_data, model_type):
    if model_type == 'nonlinear':
        return [y_data.max(), np.median(x_data)]
    return [0.5]

# ================== 拟合函数 ==================
def fit_models(df, fit_method='mean'):
    df_clean = df.dropna(subset=['Rc', 'Region_Ed', 'Region_pixels_sum'])
    if len(df_clean) < 3:
        return None

    x_data = df_clean['Rc'].values
    y_data = df_clean['Region_Ed'].values
    pixel_weights = np.sqrt(df_clean['Region_pixels_sum'].values)
    x_data = np.maximum(x_data, 1e-10)

    if fit_method == 'mean':
        unique_x = np.unique(x_data)
        mean_y = np.array([np.mean(y_data[x_data == x]) for x in unique_x])
        x_data = unique_x
        y_data = mean_y
        weights = None
    else:
        weights = np.maximum(pixel_weights, 1e-5)
        weights = weights / weights.sum() * len(weights)

    best_model = None
    best_metrics = None
    best_popt = None
    best_model_type = None
    all_models = {}

    try:
        p0 = get_initial_params(x_data, y_data, 'nonlinear')
        bounds = ([0, 0], [1, 1])
        p0 = np.clip(p0, bounds[0], bounds[1])
        popt_nl, _ = curve_fit(ed_nonlinear, x_data, y_data, p0=p0, bounds=bounds, sigma=weights, method='trf')
        y_nl = ed_nonlinear(x_data, *popt_nl)
        r2_nl = r2_score(y_data, y_nl)
        rmse_nl = np.sqrt(mean_squared_error(y_data, y_nl))
        all_models['nonlinear'] = {
            'popt': popt_nl,
            'r2': r2_nl,
            'rmse': rmse_nl,
            'y_fit': y_nl
        }
    except Exception as e:
        print(f"Nonlinear fit failed: {e}")

    try:
        popt_l, _ = curve_fit(ed_linear, x_data, y_data, p0=[1.0], bounds=(0, np.inf), sigma=weights)
        y_l = ed_linear(x_data, *popt_l)
        r2_l = r2_score(y_data, y_l)
        rmse_l = np.sqrt(mean_squared_error(y_data, y_l))
        all_models['linear'] = {
            'popt': popt_l,
            'r2': r2_l,
            'rmse': rmse_l,
            'y_fit': y_l
        }
    except Exception as e:
        print(f"Linear fit failed: {e}")

    for model_type, info in all_models.items():
        if best_model is None or info['r2'] > best_metrics['R2']:
            best_model_type = model_type
            best_model = info['y_fit']
            best_metrics = {'R2': info['r2'], 'RMSE': info['rmse']}
            best_popt = info['popt']

    return {
        'best_type': best_model_type,
        'best_popt': best_popt,
        'best_metrics': best_metrics,
        'x': x_data,
        'y': y_data,
        'all_models': all_models
    }

# ================== 绘图函数 ==================
def plot_group_fits(group_results, output_dir):
    n = len(group_results)
    cols = 3
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(cols * 5, rows * 4))

    for i, (key, model_type, popt, metrics, x_data, y_data, all_models) in enumerate(group_results):
        plt.subplot(rows, cols, i + 1)
        plt.scatter(x_data, y_data, s=20, alpha=0.6, label='Data')
        x_fit = np.linspace(min(x_data), max(x_data), 100)
        is_control = 'control' in key.lower()

        if is_control:
            if 'nonlinear' in all_models:
                y_nl = ed_nonlinear(x_fit, *all_models['nonlinear']['popt'])
                plt.plot(x_fit, y_nl, 'r-', label=f'Nonlinear (R²={all_models["nonlinear"]["r2"]:.3f})')
            if 'linear' in all_models:
                y_l = ed_linear(x_fit, *all_models['linear']['popt'])
                plt.plot(x_fit, y_l, 'b--', label=f'Linear (R²={all_models["linear"]["r2"]:.3f})')
        else:
            if model_type == 'nonlinear':
                y_fit = ed_nonlinear(x_fit, *popt)
            else:
                y_fit = ed_linear(x_fit, *popt)
            plt.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'{model_type.title()} Fit')

        plt.title(f"{key}\nR²={metrics['R2']:.3f}, RMSE={metrics['RMSE']:.3f}")
        plt.xlabel("Rc")
        plt.ylabel("Ed")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, 'Rc-Ed_group_fit_subplots.png')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Group fit subplots saved to {path}")

def plot_overall_fits(group_results, output_dir):
    plt.figure(figsize=(12, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, len(group_results)))

    for i, (key, model_type, popt, metrics, x_data, y_data, all_models) in enumerate(group_results):
        color = colors[i]
        plt.scatter(x_data, y_data, s=15, alpha=0.3, color=color, label=f"{key} Data")
        x_fit = np.linspace(min(x_data), max(x_data), 100)
        is_control = 'control' in key.lower()

        if is_control:
            if 'nonlinear' in all_models:
                y_nl = ed_nonlinear(x_fit, *all_models['nonlinear']['popt'])
                plt.plot(x_fit, y_nl, color=color, linestyle='-', linewidth=2,
                         label=f"{key} Nonlinear (R²={all_models['nonlinear']['r2']:.2f})")
            if 'linear' in all_models:
                y_l = ed_linear(x_fit, *all_models['linear']['popt'])
                plt.plot(x_fit, y_l, color=color, linestyle='--', linewidth=2,
                         label=f"{key} Linear (R²={all_models['linear']['r2']:.2f})")
        else:
            if model_type == 'nonlinear':
                y_fit = ed_nonlinear(x_fit, *popt)
            else:
                y_fit = ed_linear(x_fit, *popt)
            plt.plot(x_fit, y_fit, color=color, linewidth=2, label=f"{key} Fit (Model={model_type.title()})")

    plt.xlabel("Rc")
    plt.ylabel("Ed")
    plt.title("Combined Fits Across Groups")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    path = os.path.join(output_dir, 'Rc-Ed_overall_fit_plot.png')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Overall fit plot saved to {path}")

# ================== 主处理函数 ==================
def process_csv_with_best_model(file_path, output_path, fit_method='mean'):
    df = pd.read_csv(file_path).dropna(subset=['Rc', 'Region_Ed', 'Region_pixels_sum'])
    required = ['Metadata_hour', 'Metadata_treatment', 'Metadata_concentration', 'Rc', 'Region_Ed', 'Region_pixels_sum']
    if not all(col in df.columns for col in required):
        raise ValueError("Missing required columns")

    results = []
    group_results = []
    grouped = df.groupby(['Metadata_hour', 'Metadata_treatment', 'Metadata_concentration'])

    for (hour, treatment, conc), gdf in grouped:
        key = f'{hour}h-{treatment}-{conc}um'
        fit_result = fit_models(gdf, fit_method)
        if fit_result is None:
            continue

        result = fit_result
        best_type = result['best_type']
        best_popt = result['best_popt']
        metrics = result['best_metrics']
        x_data, y_data = result['x'], result['y']

        res = {
            'Key': key,
            'Hour': hour,
            'Treatment': treatment,
            'Concentration': conc,
            'BestModel': best_type,
            'R2': metrics['R2'],
            'RMSE': metrics['RMSE'],
            'N': len(gdf)
        }

        if 'nonlinear' in result['all_models']:
            res.update({
                'Edmax_nl': result['all_models']['nonlinear']['popt'][0],
                'Kd_nl': result['all_models']['nonlinear']['popt'][1],
                'R2_nl': result['all_models']['nonlinear']['r2'],
                'RMSE_nl': result['all_models']['nonlinear']['rmse'],
            })

        if 'linear' in result['all_models']:
            res.update({
                'a_l': result['all_models']['linear']['popt'][0],
                'R2_l': result['all_models']['linear']['r2'],
                'RMSE_l': result['all_models']['linear']['rmse'],
            })

        results.append(res)
        group_results.append((key, best_type, best_popt, metrics, x_data, y_data, result['all_models']))

    results_df = pd.DataFrame(results)
    os.makedirs(output_path, exist_ok=True)
    results_df.to_csv(os.path.join(output_path, 'Rc-Ed_fit_results.csv'), index=False)
    print(f"Fit results saved to {output_path}/Rc-Ed_fit_results.csv")

    plot_group_fits(group_results, output_path)
    plot_overall_fits(group_results, output_path)

    return results_df, group_results

# ================== 主入口函数 ==================
def main():
    folder_path = r'C:\Code\python\csv_data\gl\EGFR-GRB2实验数据\A549\20250611'
    csv_path = f"{folder_path}/rc_ed.csv"
    output_dir = r"C:/Users/pengs/Downloads"

    print("Starting model fitting and evaluation...")
    results, group_results = process_csv_with_best_model(
        csv_path,
        output_dir,
        fit_method='weighted'
    )
    print("Done.")
    print(results[['Key', 'BestModel', 'R2', 'RMSE', 'N']])

if __name__ == "__main__":
    main()
