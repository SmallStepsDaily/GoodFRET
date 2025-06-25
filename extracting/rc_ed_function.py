import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import gaussian_kde
import os

'''
rc-Region_Ed拟合算法
- 使用非线性拟合模型（edmax * Rc / (kd + Rc)）
- 加入Region_pixels_sum作为权重
- 输出每组（时间-处理-浓度）的拟合曲线与Edmax值
'''

# ================== 模型定义 ==================
def ed_nonlinear(rc, edmax, kd):
    """Nonlinear model: Ed = edmax * Rc / (kd + Rc)"""
    return edmax * rc / (kd + rc)


def get_initial_params(x_data, y_data, model_type):
    if model_type == 'nonlinear':
        return [y_data.max(), np.median(x_data)]
    return [0.5, 0.5]


# ================== 拟合函数 ==================
def fit_models(df, fit_method='mean'):
    df_clean = df.dropna(subset=['Rc', 'Region_Ed', 'Region_pixels_sum'])
    if len(df_clean) < 3:
        return None

    x_data = df_clean['Rc'].values
    y_data = df_clean['Region_Ed'].values
    pixel_weights = df_clean['Region_pixels_sum'].values
    x_data = np.maximum(x_data, 1e-10)

    if fit_method == 'mean':
        unique_x = np.unique(x_data)
        mean_y = np.array([np.mean(y_data[x_data == x]) for x in unique_x])
        x_data = unique_x
        y_data = mean_y
        weights = None
    else:
        # 1. 用出现频率作为密度（适合离散数据）
        weights = np.maximum(pixel_weights, 1e-5)  # 避免为0
        weights = weights / weights.sum() * len(weights)  # 保持 scale 稳定

    try:
        p0 = get_initial_params(x_data, y_data, 'nonlinear')
        bounds = ([0, 0], [1, 1])
        p0 = np.clip(p0, bounds[0], bounds[1])

        popt, pcov = curve_fit(ed_nonlinear, x_data, y_data, p0=p0, bounds=bounds, sigma=weights, method='trf')
        y_fit = ed_nonlinear(x_data, *popt)
        result = (popt, pcov, {
            'R2': r2_score(y_data, y_fit),
            'RMSE': np.sqrt(mean_squared_error(y_data, y_fit))
        })
        return result, x_data, y_data
    except Exception as e:
        print(f"Nonlinear fit failed: {e}")
        return None


# ================== 绘图函数 ==================
def plot_group_fits(group_results, output_dir):
    n = len(group_results)
    cols = 3
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(cols * 5, rows * 4))

    for i, (key, popt, metrics, x_data, y_data) in enumerate(group_results):
        plt.subplot(rows, cols, i + 1)
        plt.scatter(x_data, y_data, s=20, alpha=0.6, label='Data')
        x_fit = np.linspace(min(x_data), max(x_data), 100)
        y_fit = ed_nonlinear(x_fit, *popt)
        plt.plot(x_fit, y_fit, 'r-', linewidth=2, label='Nonlinear Fit')
        edmax_val = popt[0]

        plt.title(f"{key}\nR²={metrics['R2']:.3f}, RMSE={metrics['RMSE']:.3f}\nEdmax={edmax_val:.3f}")
        plt.xlabel("Rc")
        plt.ylabel("Ed")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(fontsize=8)

    plt.tight_layout()
    group_fit_path = os.path.join(output_dir, 'Rc-Ed_group_fit_subplots.png')
    plt.savefig(group_fit_path, dpi=300)
    plt.close()
    print(f"Group fit subplots saved to {group_fit_path}")


def plot_overall_fits(group_results, output_dir):
    plt.figure(figsize=(12, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, len(group_results)))

    for i, (key, popt, metrics, x_data, y_data) in enumerate(group_results):
        color = colors[i]
        plt.scatter(x_data, y_data, s=15, alpha=0.3, color=color, label=f"{key} Data")
        x_fit = np.linspace(min(x_data), max(x_data), 100)
        y_fit = ed_nonlinear(x_fit, *popt)
        edmax_val = popt[0]
        plt.plot(x_fit, y_fit, color=color, linewidth=2, label=f"{key} Fit (Edmax={edmax_val:.3f})")

    plt.xlabel("Rc")
    plt.ylabel("Ed")
    plt.title("Combined Nonlinear Fits Across Groups")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    overall_fit_path = os.path.join(output_dir, 'Rc-Ed_overall_fit_plot.png')
    plt.savefig(overall_fit_path, dpi=300)
    plt.close()
    print(f"Overall fit plot saved to {overall_fit_path}")


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
        popt, _, metrics = fit_result[0]
        x_data, y_data = fit_result[1], fit_result[2]

        results.append({
            'Key': key,
            'Hour': hour,
            'Treatment': treatment,
            'Concentration': conc,
            'Edmax': popt[0],
            'Kd': popt[1],
            'R2': metrics['R2'],
            'RMSE': metrics['RMSE'],
            'N': len(gdf)
        })

        group_results.append((key, popt, metrics, x_data, y_data))

    results_df = pd.DataFrame(results)
    os.makedirs(output_path, exist_ok=True)
    results_df.to_csv(os.path.join(output_path, 'Rc-Ed_fit_results.csv'), index=False)
    print(f"Fit results saved to {output_path}/Rc-Ed_fit_results.csv")

    plot_group_fits(group_results, output_path)
    plot_overall_fits(group_results, output_path)

    return results_df, group_results


# ================== 主入口函数 ==================
def main():
    csv_path = r"D:\data\20250513\BCLXL-BAK/rc_ed.csv"
    output_dir = r"C:/Users/pengs/Downloads"

    print("Starting model fitting and evaluation...")
    results, group_results = process_csv_with_best_model(
        csv_path,
        output_dir,
        fit_method='weighted'
    )
    print("Done.")
    print(results[['Key', 'Edmax', 'R2', 'RMSE', 'N']])


if __name__ == "__main__":
    main()
