"""
用于表型csv文件与FRET csv文件 根据元数据拼接
"""
import os
import re
import pandas as pd
from collections import defaultdict


def parse_filename(filename):
    """解析文件名，提取特征类型和组别"""
    # 正则表达式匹配{特征类型}-{组别}.csv格式
    pattern = r'^(.+)-(.+)\.csv$'
    match = re.match(pattern, filename)
    if match:
        feature_type = match.group(1)
        group = match.group(2)
        return feature_type, group
    return None, None


def normalize_group_name(group_name):
    """标准化组名，处理小时表示中的小数点差异（如4.0h → 4h）"""
    # 将类似4.0h的格式转换为4h
    return re.sub(r'(\d+)\.0h', r'\1h', group_name)


def find_matching_fret_file(group, fret_dir):
    """查找与组别匹配的FRET文件，考虑小时表示的差异"""
    normalized_group = normalize_group_name(group)

    # 首先尝试精确匹配
    exact_match = os.path.join(fret_dir, f'{group}.csv')
    if os.path.exists(exact_match):
        return exact_match

    # 尝试标准化后的匹配
    normalized_match = os.path.join(fret_dir, f'{normalized_group}.csv')
    if os.path.exists(normalized_match):
        return normalized_match

    # 尝试模糊匹配 - 比较所有FRET文件名
    for filename in os.listdir(fret_dir):
        if filename.endswith('.csv'):
            fret_group = os.path.splitext(filename)[0]
            normalized_fret_group = normalize_group_name(fret_group)
            if normalized_fret_group == normalized_group:
                return os.path.join(fret_dir, filename)

    return None


def process_phenotype_files(phenotype_dir):
    """处理表型表征值文件夹中的所有CSV文件"""
    # 按组别存储数据
    group_data = defaultdict(list)

    # 获取文件夹中所有CSV文件
    for filename in os.listdir(phenotype_dir):
        if filename.endswith('.csv') and not os.path.isdir(os.path.join(phenotype_dir, filename)):
            feature_type, group = parse_filename(filename)
            if feature_type and group:
                file_path = os.path.join(phenotype_dir, filename)
                try:
                    df = pd.read_csv(file_path)
                    # 检查是否包含'S'列
                    if 'S' in df.columns:
                        # 重命名'S'列为'{特征类型}_S'
                        df_renamed = df.rename(columns={'S': f'{feature_type}_S'})
                        group_data[group].append((feature_type, df_renamed))
                    else:
                        print(f"警告: 文件 {filename} 中不包含 'S' 列，已跳过")
                except Exception as e:
                    print(f"处理文件 {filename} 时出错: {str(e)}")
    return group_data


def merge_group_data(group_data):
    """合并同一组别的数据并计算平均值"""
    merged_results = {}

    for group, data_list in group_data.items():
        if not data_list:
            continue

        # 获取元数据列（取第一个数据框的列作为基准）
        first_feature_type, first_df = data_list[0]
        meta_columns = [col for col in first_df.columns if col.startswith('Metadata_') or col == 'ObjectNumber']

        # 初始化合并结果为第一个数据框的元数据和第一个特征列
        first_feature_col = f'{first_feature_type}_S'
        merged_df = first_df[meta_columns + [first_feature_col]].copy()

        # 收集所有特征列名，先添加第一个特征
        feature_columns = [first_feature_col]

        # 合并各个特征类型的数据（从第二个开始）
        for feature_type, df in data_list[1:]:
            # 只保留元数据列和当前特征列，确保行数匹配
            current_feature_col = f'{feature_type}_S'
            current_columns = meta_columns + [current_feature_col]

            if all(col in df.columns for col in current_columns):
                # 合并数据
                merged_df = merged_df.merge(df[current_columns], on=meta_columns, how='inner')
                feature_columns.append(current_feature_col)
            else:
                print(f"警告: 组别 {group} 的 {feature_type} 数据缺少必要的元数据列或特征列，已跳过")

        # 计算'S'列（所有特征列的平均值）
        if feature_columns:
            merged_df['S'] = merged_df[feature_columns].mean(axis=1)
            merged_results[group] = merged_df
        else:
            print(f"警告: 组别 {group} 没有可用的特征数据，已跳过")

    return merged_results


def match_fret_data(merged_data, fret_dir):
    """匹配FRET表征值数据，使用改进的匹配规则"""
    result_with_fret = {}

    for group, df in merged_data.items():
        # 使用新的匹配函数查找FRET文件
        fret_file = find_matching_fret_file(group, fret_dir)

        if fret_file:
            try:
                fret_df = pd.read_csv(fret_file)

                # 确定用于匹配的元数据列（取两个数据框共有的元数据列）
                meta_columns = [col for col in df.columns if col.startswith('Metadata_') or col == 'ObjectNumber']
                fret_meta_columns = [col for col in fret_df.columns if
                                     col.startswith('Metadata_') or col == 'ObjectNumber']
                common_meta = list(set(meta_columns) & set(fret_meta_columns))

                if not common_meta:
                    print(f"警告: 组别 {group} 没有共同的元数据列，无法匹配FRET数据")
                    result_with_fret[group] = df
                    continue

                # 按共同元数据列合并
                if 'E' in fret_df.columns:
                    # 只保留需要的列
                    fret_df_filtered = fret_df[common_meta + ['E']]
                    merged_with_fret = df.merge(fret_df_filtered, on=common_meta, how='left')
                    result_with_fret[group] = merged_with_fret
                    print(f"已匹配组别 {group} 的FRET文件: {os.path.basename(fret_file)}")
                else:
                    print(f"警告: FRET文件 {os.path.basename(fret_file)} 中不包含 'E' 列")
                    result_with_fret[group] = df
            except Exception as e:
                print(f"处理FRET文件 {os.path.basename(fret_file)} 时出错: {str(e)}")
                result_with_fret[group] = df
        else:
            print(f"警告: 未找到组别 {group} 的FRET文件，已跳过匹配")
            result_with_fret[group] = df

    return result_with_fret


def save_results(results, output_dir):
    """保存结果到输出文件夹"""
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    for group, df in results.items():
        output_file = os.path.join(output_dir, f'{group}.csv')
        try:
            df.to_csv(output_file, index=False)
            print(f"已保存结果到 {output_file}")
        except Exception as e:
            print(f"保存文件 {output_file} 时出错: {str(e)}")


def main(phenotype_dir, fret_dir, output_dir):
    """主函数：协调处理流程"""
    print("开始处理表型表征值文件...")
    group_data = process_phenotype_files(phenotype_dir)

    print("合并组别数据并计算平均值...")
    merged_data = merge_group_data(group_data)

    print("匹配FRET表征值数据...")
    final_results = match_fret_data(merged_data, fret_dir)

    print("保存结果...")
    save_results(final_results, output_dir)

    print("处理完成！")


if __name__ == "__main__":
    # 示例用法
    phenotype_directory = r'C:\Code\python\csv_data\qrm\20250714\20250709\转录表型表征值'  # 表型表征值文件夹路径
    fret_directory = r'C:\Code\python\csv_data\qrm\20250714\20250709\Foxo3a表征值'  # FRET表征值文件夹路径
    output_directory = r'C:\Code\python\csv_data\qrm\20250714\20250709\单细胞表征值'  # 输出文件夹路径

    main(phenotype_directory, fret_directory, output_directory)
