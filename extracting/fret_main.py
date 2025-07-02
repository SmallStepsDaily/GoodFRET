from extracting.rc_ed_function import process_csv_with_best_model
from extracting.rc_ed_phenotype import merge_feature_files
from extracting.rc_ed_single_cell_judgment import analyze_fret_data

def main(folder_path, model_dict, weight, need_rc_ed, ed_fp_feature_name):

    # 拟合rc_ed曲线
    csv_path = f"{folder_path}/rc_ed.csv"
    print("Starting model fitting and evaluation...")
    results, group_results = process_csv_with_best_model(
        csv_path,
        folder_path,
        fit_method='weighted'
    )
    print("Done.")

    # 判断是否发生了FRET变化
    input_csv = f"{folder_path}\FRET.csv"  # 替换为你的 CSV 文件路径
    analyze_fret_data(input_csv, model_dict=model_dict, output_path=folder_path, save_plot=True)


    # 合并表型表征值结果
    PHENOTYPE_DIR = f"{folder_path}\表型表征值"  # 表型特征文件目录
    FRET_FILE = f"{folder_path}\Rc-Ed_FRET_analyzed.csv"  # FRET特征文件路径
    OUTPUT_DIR = f"{folder_path}\单细胞匹配"  # 输出目录

    # 运行合并操作
    results = merge_feature_files(
        phenotype_dir=PHENOTYPE_DIR,
        fret_file=FRET_FILE,
        output_dir=OUTPUT_DIR,
        weight=weight,
        need_rc_ed=need_rc_ed,
        ed_fp_feature_name=ed_fp_feature_name
    )

    # 打印结果摘要
    if results:
        print(f"\n成功处理 {len(results)} 个文件")
    else:
        print("\n未处理任何文件或处理过程中出错")

if __name__ == '__main__':
    # need_rc_ed True表示需要使用rc-ed拟合曲线计算得出的效率表征值
    # need_rc_ed False表示需要只需要使用踩点所获得的效率计算效率表征值


    # main(r'C:\Code\python\csv_data\gl\BCLXL-BAX实验数据\20250513', 0.5, True, 'Fp_region_PCC')

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
    main(r'C:\Code\python\csv_data\gl\EGFR-GRB2实验数据\A549\20250618', EGFR_model_exprs, 0.5, True, 'Fp_region_PCC')