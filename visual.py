import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import umap

# 1️⃣  加载数据
embeddings = np.load("test_embeddings.npy")  # (num_samples, feature_dim)
labels = np.load("test_labels.npy")  # (num_samples,)

# 2️⃣  创建 DataFrame
data = pd.DataFrame(embeddings)
data['label'] = labels  # 添加标签列

# 3️⃣  选择出现次数最多的 6 个标签
top_labels = data['label'].value_counts().index[:6]  # 获取出现最多的 6 个标签
filtered_data = data[data['label'].isin(top_labels)].copy()  # 显式复制，避免 SettingWithCopyWarning

# 4️⃣  PCA 降维到 50 维
pca = PCA(n_components=50)
pca_result = pca.fit_transform(filtered_data.drop(columns=['label']))

# 5️⃣  使用 UMAP 降维到 2 维
umap_model = umap.UMAP(n_components=3, n_neighbors=50, min_dist=0.1, metric='manhattan')
umap_result = umap_model.fit_transform(pca_result)

# 6️⃣  创建 UMAP 结果的新列（使用 .loc 避免警告）
filtered_data.loc[:, 'umap-1'] = umap_result[:, 0]
filtered_data.loc[:, 'umap-2'] = umap_result[:, 1]

# 7️⃣  可视化 UMAP 结果
plt.figure(figsize=(10, 8))
sns.scatterplot(x='umap-1', y='umap-2', hue='label', data=filtered_data, palette='Set1', s=100)
# plt.legend(title='Labels', loc='upper right')

# 保存高质量图片
plt.savefig("umap_visualization.pdf", format="pdf", dpi=300, bbox_inches='tight')
plt.savefig("umap_visualization.svg", format="svg", dpi=300, bbox_inches='tight')
plt.savefig("umap_visualization.eps", format="eps", dpi=300, bbox_inches='tight')  # 适用于 LaTeX
plt.savefig("umap_visualization.png", format="png", dpi=600, bbox_inches='tight')  # 高分辨率 PNG

plt.show()
