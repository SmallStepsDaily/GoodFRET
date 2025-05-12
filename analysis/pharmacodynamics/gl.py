import numpy as np
import pandas as pd


# 读入excel 或者csv 格式数据
# TODO 待实现
# 输入的数据
# treatments = np.array(['control', 'control', 'A1331852', 'A1331852', 'ABT-199', 'ABT-199', 'Trypsin', 'Trypsin', 'NACL', 'NACL'])
# times = np.array([5, 11, 5, 11, 5, 11, 1, 2, 1, 2])
# concentrations = np.array([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 0.6, 0.6])
# S = np.array([0, 0, 0.1568, 0.1566, 0.0752, 0.1303, 0.1540, 0.1540, 0.1996, 0.2115])
# E = np.array([0, 0, 0.3431, 0.3146, 0.0735, 0.0258, 0.0318, 0.0318, 0.0218, 0.0668])


def gaolu_function(df):
    result_str = ''
    treatments = df['Metadata_treatment'].values
    # 时间进行归一化，假设24小时为最大的时间
    times = df['Metadata_hour'].values / 24
    # 浓度单位默认时 μm 为单位，乘上10-6进行转化
    concentrations = df['Metadata_concentration'].values * 1e-6
    S = df['S'].values
    E = df['E'].values

    # 约定限定条件 alpha + beta = 1
    alpha_list = np.linspace(0, 1, 1000)
    beta_list = 1 - alpha_list

    unique_treatments = np.unique(treatments)
    # 第一个循环，计算每组的平均值average_Z
    for treatment in unique_treatments:
        # 寻找对应的数据，为布尔掩码
        treatment_mask = (treatment == treatments)
        treatment_time = times[treatment_mask]
        treatment_S = S[treatment_mask]
        treatment_E = E[treatment_mask]
        treatment_concentration = concentrations[treatment_mask]
        ###########################################
        # 计算矫正因子  alpha 和 beta
        ###########################################
        # 创建一个空列表来存储每个alpha-beta组合的结果
        treatment_R_list = []
        for alpha, beta in zip(alpha_list, beta_list):
            # 对于每个alpha-beta组合，计算修正后的响应值
            R = (alpha * treatment_E + 1) ** (beta * treatment_S)
            treatment_R_list.append(R)

        # 将列表转换为NumPy数组，并确保它的形状正确（样本数 x alpha-beta组合数）
        treatment_R = np.array(treatment_R_list).T
        # 计算每列（即每个alpha-beta组合）的标准差
        stds = treatment_R.std(axis=0)
        # 找到最大标准差的索引
        max_std_index = np.argmax(stds)
        # 得出矫正因子 alpha 和 beta
        alpha = alpha_list[max_std_index]
        beta = beta_list[max_std_index]
        if alpha == 0:
            alpha = 0.5
            beta = 0.5
        result_str += f"{treatment} =============> alpha:{alpha}, beta: {beta}\n"
        ##################################################
        # 计算不同时间和浓度梯度下的药效矩阵 R
        ##################################################
        R_matrix = 100* ((alpha * treatment_E + 1) ** (beta * treatment_S) - 1)
        result_str += f"{treatment} =============> R_matrix: {R_matrix}\n"
        #################################################
        # 计算权重函数矩阵 B
        #################################################
        B_matrix = (-np.log10(treatment_concentration + 1e-10)) ** (-treatment_time)
        B_matrix = B_matrix / (1 + B_matrix)
        result_str += f"{treatment} =============> B_matrix: {B_matrix}\n"
        ##################################################
        # 显示最后的结果取值
        ##################################################
        result_str += f"{treatment} 组的药效值 {np.sum(R_matrix * B_matrix)}\n"

    return result_str

if __name__ == "__main__":
    data = pd.read_csv(r"C:\Code\python\csv_data\hql\20250430\药效评估\药物评价.csv", encoding='GBK')
    print(gaolu_function(data))