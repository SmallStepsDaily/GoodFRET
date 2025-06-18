import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os


# 定义非线性拟合函数
def ed_nonlinear(rc, edmax, kd):
    """非线性模型: E_D = E_{Dmax} × R_c / (K_d + R_c)"""
    return edmax * rc / (kd + rc)


# 定义线性拟合函数（过原点）
def ed_linear(rc, slope):
    """线性模型: E_D = slope × R_c (过原点)"""
    return slope * rc


# 拟合函数（支持非线性和线性模型）
def fit_models(df, use_weighted=False):
    """
    对数据进行非线性和线性拟合，并比较模型性能

    Returns:
    - 非线性拟合结果: (popt, pcov, metrics) 或 None
    - 线性拟合结果: (popt, pcov, metrics) 或 None
    """
    df_clean = df.dropna(subset=['Rc', 'Ed'])

    if len(df_clean) < 3:
        return None, None

    x_data = df_clean['Rc'].values
    y_data = df_clean['Ed'].values

    # 处理RC=0的情况
    if np.any(x_data == 0):
        x_data = np.maximum(x_data, 1e-10)

    # 非线性拟合
    nonlin_result = None
    try:
        nonlin_p0 = [0.5, 0.5]
        nonlin_bounds = ([0, 0], [1, 1])
        nonlin_weights = None

        if use_weighted:
            unique_rc, counts = np.unique(x_data, return_counts=True)
            rc_density = dict(zip(unique_rc, counts))
            nonlin_weights = np.array([1 / rc_density[rc] for rc in x_data])

        nonlin_popt, nonlin_pcov = curve_fit(
            ed_nonlinear, x_data, y_data,
            p0=nonlin_p0, bounds=nonlin_bounds, sigma=nonlin_weights
        )

        nonlin_y_fit = ed_nonlinear(x_data, *nonlin_popt)
        nonlin_residuals = y_data - nonlin_y_fit
        nonlin_ss_res = np.sum(nonlin_residuals ** 2)
        nonlin_ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        nonlin_r2 = 1 - (nonlin_ss_res / nonlin_ss_tot)
        nonlin_rmse = np.sqrt(nonlin_ss_res / len(y_data))

        nonlin_result = (nonlin_popt, nonlin_pcov, {'R2': nonlin_r2, 'RMSE': nonlin_rmse})
    except Exception as e:
        print(f"非线性拟合失败: {str(e)}")

    # 线性拟合（过原点）
    lin_result = None
    try:
        lin_p0 = [0.5]
        lin_bounds = ([0], [1])  # slope范围0-1

        lin_popt, lin_pcov = curve_fit(
            ed_linear, x_data, y_data,
            p0=lin_p0, bounds=lin_bounds
        )

        lin_y_fit = ed_linear(x_data, *lin_popt)
        lin_residuals = y_data - lin_y_fit
        lin_ss_res = np.sum(lin_residuals ** 2)
        lin_ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        lin_r2 = 1 - (lin_ss_res / lin_ss_tot)
        lin_rmse = np.sqrt(lin_ss_res / len(y_data))

        lin_result = (lin_popt, lin_pcov, {'R2': lin_r2, 'RMSE': lin_rmse})
    except Exception as e:
        print(f"线性拟合失败: {str(e)}")

    return nonlin_result, lin_result


# 绘制所有组的最优拟合曲线（只显示每个组的最佳模型）
def plot_best_fits(group_results, output_path='best_fit_plot.png'):
    """绘制每个组的最优拟合模型，减少图像混乱"""
    plt.figure(figsize=(12, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(group_results))))

    # 分离control组和其他组
    control_groups = []
    other_groups = []

    for i, (group_key, best_model, best_popt, best_metrics, x_data, y_data) in enumerate(group_results):
        is_control = 'control' in group_key.lower()
        group = (i, group_key, best_model, best_popt, best_metrics, x_data, y_data)
        (control_groups if is_control else other_groups).append(group)

    # 绘制control组，再绘制其他组
    for i, (_, group_key, model, popt, metrics, x_data, y_data) in enumerate(control_groups + other_groups):
        color = colors[i % len(colors)]

        # 绘制原始数据点
        plt.scatter(x_data, y_data, s=15, color=color, alpha=0.4, label=f'{group_key} (Data)')

        # 绘制最优模型
        x_fit = np.linspace(min(x_data), max(x_data), 100)
        if model == "nonlinear":
            edmax, kd = popt
            y_fit = ed_nonlinear(x_fit, edmax, kd)
            plt.plot(x_fit, y_fit, color=color, linewidth=2,
                     label=f'Nonlinear: {group_key}, Edmax={edmax:.3f}, Kd={kd:.3f}')
        elif model == "linear":
            slope = popt[0]
            y_fit = ed_linear(x_fit, slope)
            plt.plot(x_fit, y_fit, color=color, linewidth=2,
                     label=f'Linear: {group_key}, slope={slope:.3f}')

    plt.xlabel('Rc')
    plt.ylabel('Ed')
    plt.title('Ed-Rc Fitting: Best Model for Each Group')
    plt.grid(True, linestyle='--', alpha=0.7)

    # 优化图例：分离control和其他组
    handles, labels = plt.gca().get_legend_handles_labels()
    control_handles, control_labels = [], []
    other_handles, other_labels = [], []

    for h, l in zip(handles, labels):
        if 'control' in l.lower():
            control_handles.append(h)
            control_labels.append(l)
        else:
            other_handles.append(h)
            other_labels.append(l)

    # 合并图例：control在前，其他在后
    plt.legend(control_handles + other_handles, control_labels + other_labels,
               loc='best', ncol=1, fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"最优拟合图已保存至 {output_path}")


# 主函数：处理CSV并选择最优模型
def process_csv_with_best_model(file_path, output_path, use_weighted=True):
    """处理CSV并为每个组选择最优拟合模型"""
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['Rc', 'Ed'])
    required_columns = ['Metadata_hour', 'Metadata_treatment', 'Metadata_concentration', 'Rc', 'Ed']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"缺少必要列: {', '.join(missing_columns)}")

    results = []
    group_results = []  # 存储用于绘图的数据（组名、最优模型、参数等）

    grouped = df.groupby(['Metadata_hour', 'Metadata_treatment', 'Metadata_concentration'])

    for (hour, treatment, conc), group_df in grouped:
        key = f'{hour}h-{treatment}-{conc}um'
        print(f"处理组: {key} (n={len(group_df)})")

        # 执行两种模型的拟合
        nonlin_result, lin_result = fit_models(group_df, use_weighted=use_weighted)

        # 选择最优模型
        best_model = None
        best_popt = None
        best_metrics = None

        if nonlin_result and lin_result:
            # 比较两种模型
            nonlin_r2, nonlin_rmse = nonlin_result[2]['R2'], nonlin_result[2]['RMSE']
            lin_r2, lin_rmse = lin_result[2]['R2'], lin_result[2]['RMSE']

            # R²越高且RMSE越低，模型越好
            nonlin_score = nonlin_r2 - nonlin_rmse
            lin_score = lin_r2 - lin_rmse

            if nonlin_score > lin_score:
                best_model = "nonlinear"
                best_popt, _, best_metrics = nonlin_result
            else:
                best_model = "linear"
                best_popt, _, best_metrics = lin_result
        elif nonlin_result:
            best_model = "nonlinear"
            best_popt, _, best_metrics = nonlin_result
        elif lin_result:
            best_model = "linear"
            best_popt, _, best_metrics = lin_result
        else:
            print(f"两组模型均拟合失败: {key}")
            continue

        # 保存结果
        results.append({
            'Key': key,
            'Metadata_hour': hour,
            'Metadata_treatment': treatment,
            'Metadata_concentration': conc,
            'Best_Model': best_model,
            # 非线性模型参数（如果是最优模型）
            'Nonlin_Edmax': nonlin_result[0][0] if (best_model == "nonlinear" and nonlin_result) else np.nan,
            'Nonlin_Kd': nonlin_result[0][1] if (best_model == "nonlinear" and nonlin_result) else np.nan,
            'Nonlin_R2': nonlin_result[2]['R2'] if nonlin_result else np.nan,
            'Nonlin_RMSE': nonlin_result[2]['RMSE'] if nonlin_result else np.nan,
            # 线性模型参数（如果是最优模型）
            'Lin_Slope': lin_result[0][0] if (best_model == "linear" and lin_result) else np.nan,
            'Lin_R2': lin_result[2]['R2'] if lin_result else np.nan,
            'Lin_RMSE': lin_result[2]['RMSE'] if lin_result else np.nan,
            'Data_points': len(group_df)
        })

        # 保存绘图数据
        group_results.append((
            key,
            best_model,
            best_popt,
            best_metrics,
            group_df['Rc'].values,
            group_df['Ed'].values
        ))

    # 绘制最优拟合图
    if group_results:
        plot_best_fits(group_results, f'{output_path}/best_fit_plot.png')

    # 保存结果到CSV
    if results:
        results_df = pd.DataFrame(results)
        os.makedirs(output_path, exist_ok=True)
        results_df.to_csv(f'{output_path}/best_fitting_results.csv', index=False)
        print(f"结果已保存至 {output_path}/best_fitting_results.csv")
        return results_df

    return None


# 主程序入口
if __name__ == "__main__":
    csv_file = r'C:\Code\python\csv_data\gl\20250513\BCLXL-BAK/rc_ed.csv'  # 替换为实际文件路径
    output_path = r"C:\Users\pengs\Downloads"  # 替换为输出路径

    results = process_csv_with_best_model(csv_file, output_path, use_weighted=True)

    if results is not None and not results.empty:
        print("\n=== 最优模型摘要 ===")
        print(results[['Key', 'Best_Model', 'Nonlin_R2', 'Lin_R2', 'Data_points']])
    else:
        print("未获得有效拟合结果。")