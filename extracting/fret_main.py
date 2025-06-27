from extracting.rc_ed_function import process_csv_with_best_model
from extracting.rc_ed_phenotype import merge_feature_files
from extracting.rc_ed_single_cell_judgment import analyze_fret_data

def main(folder_path):

    # 拟合rc_ed曲线
    csv_path = f"{folder_path}/rc_ed.csv"
    print("Starting model fitting and evaluation...")
    results, group_results = process_csv_with_best_model(
        csv_path,
        folder_path,
        fit_method='weighted'
    )
    print("Done.")
    print(results[['Key', 'Edmax', 'R2', 'RMSE', 'N']])

    # 判断是否发生了FRET变化
    input_csv = f"{folder_path}\FRET.csv"  # 替换为你的 CSV 文件路径
    analyze_fret_data(input_csv, save_plot=True)


    # 合并表型表征值结果
    PHENOTYPE_DIR = f"{folder_path}\表型表征值"  # 表型特征文件目录
    FRET_FILE = f"{folder_path}\Rc-Ed_FRET_analyzed.csv"  # FRET特征文件路径
    OUTPUT_DIR = f"{folder_path}\单细胞匹配"  # 输出目录

    # 运行合并操作
    results = merge_feature_files(
        phenotype_dir=PHENOTYPE_DIR,
        fret_file=FRET_FILE,
        output_dir=OUTPUT_DIR
    )

    # 打印结果摘要
    if results:
        print(f"\n成功处理 {len(results)} 个文件")
    else:
        print("\n未处理任何文件或处理过程中出错")

if __name__ == '__main__':
    main(r'C:\Code\python\csv_data\gl\BCLXL-BAX实验数据\20250523')