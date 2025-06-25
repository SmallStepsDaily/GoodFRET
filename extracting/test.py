import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

# 生成示例数据（双峰分布）
np.random.seed(42)
x1 = np.random.normal(-2, 0.5, 500)  # 左侧密集区域
x2 = np.random.normal(2, 1.0, 100)   # 右侧稀疏区域
x_data = np.concatenate([x1, x2])

# 计算密度和权重
kde = gaussian_kde(x_data)
densities = np.maximum(kde(x_data), 1e-10)
weights = densities / densities.sum() * len(x_data)

# 可视化结果
plt.figure(figsize=(12, 6))

# 1. 数据分布直方图
plt.subplot(121)
plt.hist(x_data, bins=50, alpha=0.7, color='blue')
plt.title('数据分布')
plt.xlabel('x值')
plt.ylabel('频数')

# 2. 权重分布
plt.subplot(122)
plt.scatter(x_data, weights, alpha=0.5, s=20, c='red')
plt.axhline(y=1, color='k', linestyle='--', alpha=0.5)
plt.title('生成的权重（密度高的点权重更大）')
plt.xlabel('x值')
plt.ylabel('权重')

plt.tight_layout()
plt.show()

# 打印统计信息
print(f"数据点总数: {len(x_data)}")
print(f"权重总和: {np.sum(weights):.2f}")  # 应接近 len(x_data)
print(f"最大权重: {np.max(weights):.2f}")  # 左侧密集区域的点权重较大
print(f"最小权重: {np.min(weights):.6f}")  # 右侧稀疏区域的点权重较小