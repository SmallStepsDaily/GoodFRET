import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import gaussian_kde
import os

'''
rc-ed拟合算法
1. 可以使用线性拟合和非线性拟合
2. 输出{时间}-{干扰}-{浓度}的曲线图像

待改进部分：
1. 数据筛选，筛选不合理的数据
'''

# ================== Model definitions ==================
def ed_nonlinear(rc, edmax, kd):
    """Nonlinear model: Ed = edmax * Rc / (kd + Rc)"""
    return edmax * rc / (kd + rc)


def ed_linear(rc, slope, intercept=0):
    """Linear model: Ed = slope * Rc + intercept"""
    return slope * rc + intercept


def get_initial_params(x_data, y_data, model_type):
    if model_type == 'nonlinear':
        return [y_data.max(), np.median(x_data)]
    elif model_type == 'linear':
        slope, intercept, *_ = stats.linregress(x_data, y_data)
        return [slope, intercept]
    return [0.5, 0.5]


# ================== Fitting function ==================
def fit_models(df, fit_method='mean', use_intercept=False, force_nonlinear=False):
    df_clean = df.dropna(subset=['Rc', 'Ed'])
    if len(df_clean) < 3:
        return None, None

    x_data = df_clean['Rc'].values
    y_data = df_clean['Ed'].values
    x_data = np.maximum(x_data, 1e-10)

    if fit_method == 'mean':
        unique_x = np.unique(x_data)
        mean_y = np.array([np.mean(y_data[x_data == x]) for x in unique_x])
        x_data = unique_x
        y_data = mean_y

    # Nonlinear fit
    nonlin_result = None
    try:
        p0 = get_initial_params(x_data, y_data, 'nonlinear')
        bounds = ([0, 0], [1, 1])
        p0 = np.clip(p0, bounds[0], bounds[1])

        weights = None
        if fit_method == 'weighted':
            kde = gaussian_kde(x_data)
            densities = np.maximum(kde(x_data), 1e-10)
            weights = densities / densities.sum() * len(densities)

        popt, pcov = curve_fit(ed_nonlinear, x_data, y_data, p0=p0, bounds=bounds, sigma=weights)
        y_fit = ed_nonlinear(x_data, *popt)
        nonlin_result = (popt, pcov, {
            'R2': r2_score(y_data, y_fit),
            'RMSE': np.sqrt(mean_squared_error(y_data, y_fit))
        })
    except Exception as e:
        print(f"Nonlinear fit failed: {e}")

    # Linear fit
    lin_result = None
    if not force_nonlinear:
        try:
            if use_intercept:
                p0 = get_initial_params(x_data, y_data, 'linear')
                bounds = ([0, -1], [1, 1])
                func = ed_linear
            else:
                p0 = [get_initial_params(x_data, y_data, 'linear')[0]]
                bounds = ([0], [1])
                func = lambda rc, slope: ed_linear(rc, slope, intercept=0)

            popt, pcov = curve_fit(func, x_data, y_data, p0=p0, bounds=bounds)
            y_fit = func(x_data, *popt)
            lin_result = (popt, pcov, {
                'R2': r2_score(y_data, y_fit),
                'RMSE': np.sqrt(mean_squared_error(y_data, y_fit))
            })
        except Exception as e:
            print(f"Linear fit failed: {e}")

    return nonlin_result, lin_result


# ================== Plotting functions ==================
def plot_group_fits(group_results, output_dir):
    """Plot individual group subplots: scatter + nonlinear fit curve with Edmax value in title"""
    n = len(group_results)
    cols = 3
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(cols * 5, rows * 4))

    for i, (key, model, popt, metrics, x_data, y_data) in enumerate(group_results):
        plt.subplot(rows, cols, i + 1)
        plt.scatter(x_data, y_data, s=20, alpha=0.6, label='Data')

        if model == 'nonlinear':
            x_fit = np.linspace(min(x_data), max(x_data), 100)
            y_fit = ed_nonlinear(x_fit, *popt)
            plt.plot(x_fit, y_fit, 'r-', linewidth=2, label='Nonlinear Fit')
            edmax_val = popt[0]
        else:
            edmax_val = np.nan

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
    """Plot all groups combined: all scatter points + nonlinear fit curves with Edmax values in legend"""
    plt.figure(figsize=(12, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, len(group_results)))

    for i, (key, model, popt, metrics, x_data, y_data) in enumerate(group_results):
        color = colors[i]
        plt.scatter(x_data, y_data, s=15, alpha=0.3, color=color, label=f"{key} Data")
        if model == 'nonlinear':
            x_fit = np.linspace(min(x_data), max(x_data), 100)
            y_fit = ed_nonlinear(x_fit, *popt)
            edmax_val = popt[0]
            plt.plot(x_fit, y_fit, color=color, linewidth=2, label=f"{key} Fit (Edmax={edmax_val:.3f})")
        else:
            plt.plot([], [], color=color, label=f"{key} Fit (No nonlinear)")

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


# ================== Main processing function ==================
def process_csv_with_best_model(file_path, output_path, fit_method='mean', use_intercept=False, force_nonlinear=False):
    df = pd.read_csv(file_path).dropna(subset=['Rc', 'Ed'])
    required = ['Metadata_hour', 'Metadata_treatment', 'Metadata_concentration', 'Rc', 'Ed']
    if not all(col in df.columns for col in required):
        raise ValueError("Missing required columns")

    results = []
    group_results = []
    grouped = df.groupby(['Metadata_hour', 'Metadata_treatment', 'Metadata_concentration'])

    for (hour, treatment, conc), gdf in grouped:
        key = f'{hour}h-{treatment}-{conc}um'
        nonlin, lin = fit_models(gdf, fit_method, use_intercept, force_nonlinear)

        if force_nonlinear and nonlin:
            best_model = "nonlinear"
            popt, _, metrics = nonlin
        elif not force_nonlinear:
            if nonlin and lin:
                n = len(gdf)
                aic_nonlin = n * np.log(nonlin[2]['RMSE']) + 4
                aic_lin = n * np.log(lin[2]['RMSE']) + (4 if use_intercept else 2)
                best_model = 'nonlinear' if aic_nonlin < aic_lin else 'linear'
                popt, _, metrics = nonlin if best_model == 'nonlinear' else lin
            elif nonlin:
                best_model, popt, _, metrics = 'nonlinear', *nonlin
            elif lin:
                best_model, popt, _, metrics = 'linear', *lin
            else:
                continue
        else:
            continue

        results.append({
            'Key': key,
            'Hour': hour,
            'Treatment': treatment,
            'Concentration': conc,
            'Best_Model': best_model,
            'Edmax': popt[0] if best_model == 'nonlinear' else np.nan,
            'Kd': popt[1] if best_model == 'nonlinear' else np.nan,
            'Slope': popt[0] if best_model == 'linear' else np.nan,
            'Intercept': popt[1] if (use_intercept and best_model == 'linear') else 0,
            'R2': metrics['R2'],
            'RMSE': metrics['RMSE'],
            'N': len(gdf)
        })

        group_results.append((key, best_model, popt, metrics, gdf['Rc'].values, gdf['Ed'].values))

    results_df = pd.DataFrame(results)
    os.makedirs(output_path, exist_ok=True)
    results_df.to_csv(os.path.join(output_path, 'Rc-Ed_fit_results.csv'), index=False)
    print(f"Fit results saved to {output_path}/fit_results.csv")

    # Plotting
    plot_group_fits(group_results, output_path)
    plot_overall_fits(group_results, output_path)

    return results_df, group_results


# ================== Main entry point ==================
def main():
    csv_path = r"C:\Code\python\csv_data\gl\20250515\BCLXL-BAK/rc_ed.csv"
    output_dir = r"C:/Users/pengs/Downloads"

    print("Starting model fitting and evaluation...")
    results, group_results = process_csv_with_best_model(
        csv_path,
        output_dir,
        fit_method='weighted',
        use_intercept=False,
        force_nonlinear=True
    )
    print("Done.")
    print(results[['Key', 'Best_Model', 'R2', 'RMSE', 'N']])


if __name__ == "__main__":
    main()
