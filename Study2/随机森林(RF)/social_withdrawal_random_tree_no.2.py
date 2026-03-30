
# 导入所需的库
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error


# 设置更美观的样式
sns.set(style="whitegrid")

# 加载数据
df_train = pd.read_csv('part.csv', encoding='utf-8')
df_predict = pd.read_csv('full.csv', encoding='utf-8')

# 如果小数点使用逗号分隔
df_train = pd.read_csv('part.csv', encoding='utf-8', decimal=',')
df_predict = pd.read_csv('full.csv', encoding='utf-8', decimal=',')

# 检查数据结构
print(df_train.head())
print(df_predict.head())

# 特征提取：使用文本数据创建TF-IDF矩阵
# 使用微博文本和分词后的文本组合
vectorizer = TfidfVectorizer(max_features=1000)
X_train_text = vectorizer.fit_transform(df_train['weibo_text'] + " " + df_train['tokenized_text']).toarray()

# 目标变量：异性社交回避得分
y_avoidance_train = df_train['opposite_sex_social_avoidance']

# 对预测数据集进行特征提取
X_predict_text = vectorizer.transform(df_predict['weibo_text'] + " " + df_predict['tokenized_text']).toarray()

# 初始化随机森林回归器
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# 训练模型
rf_model.fit(X_train_text, y_avoidance_train)

# 对训练集进行预测
y_train_pred = rf_model.predict(X_train_text)

# 对新数据进行预测
df_predict['predicted_opposite_sex_social_avoidance'] = rf_model.predict(X_predict_text)

# 保存预测结果到CSV文件，并使用 'utf-8-sig' 编码避免乱码
output_file = 'predicted_scores_rf.csv'
df_predict.to_csv(output_file, encoding='utf-8-sig', index=False)

print(f'预测结果已保存到文件: {output_file}')
print(df_predict.head())  # 检查前几行结果，确保无误

# 计算模型的MSE和R²
mse_rf = mean_squared_error(y_avoidance_train, y_train_pred)
r2_rf = r2_score(y_avoidance_train, y_train_pred)

print(f'MSE for Random Forest: {mse_rf}')
print(f'R² for Random Forest: {r2_rf}')

# 如果需要进一步计算RMSE
rmse_rf = np.sqrt(mse_rf)
print(f'RMSE for Random Forest: {rmse_rf}')

# 计算 MAE
mae_rf = mean_absolute_error(y_avoidance_train, y_train_pred)
print(f'MAE for Random Forest: {mae_rf}')


# 散点图：实际值 vs 预测值
plt.figure(figsize=(10, 8))  # 调整图像大小
plt.scatter(y_avoidance_train, y_train_pred, color='blue', alpha=0.6, edgecolor='k', s=60)  # 调整透明度、边缘颜色

# 对角线：表示预测值与真实值完全相等的情况
plt.plot([y_avoidance_train.min(), y_avoidance_train.max()], [y_avoidance_train.min(), y_avoidance_train.max()], 'k--', lw=2)

# 添加横坐标和纵坐标的注释
plt.xlabel('真实值（Actual Values）', fontsize=14)
plt.ylabel('预测值（Predicted Values）', fontsize=14)

# 移除图例
 plt.legend(loc=None) 

# 显示网格
plt.grid(True)

# 显示图像
plt.show()

# 计算残差（实际值 - 预测值）
residuals = y_avoidance_train - y_train_pred

# 绘制残差图：预测值 vs 残差
plt.figure(figsize=(10, 8))  # 调整图像大小
plt.scatter(y_train_pred, residuals, color='blue', alpha=0.6, edgecolor='k', s=60)  # 绘制散点图，预测值 vs 残差
plt.axhline(y=0, color='k', linestyle='--', lw=2)  # 添加水平线表示残差为 0 的位置
# 添加横坐标和纵坐标的注释
plt.xlabel('预测值（Predicted Values）', fontsize=14)
plt.ylabel('残差（Residuals）', fontsize=14)
plt.title('残差图（Residual Plot）', fontsize=16)
plt.grid(True)
plt.show()











import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error

# 设置更美观的样式
sns.set(style="whitegrid")

# 加载数据
df_train = pd.read_csv('part.csv', encoding='utf-8')
df_predict = pd.read_csv('full.csv', encoding='utf-8')

# 如果小数点使用逗号分隔
df_train = pd.read_csv('part.csv', encoding='utf-8', decimal=',')
df_predict = pd.read_csv('full.csv', encoding='utf-8', decimal=',')

# 检查数据结构
print(df_train.head())
print(df_predict.head())

# 特征提取：使用文本数据创建TF-IDF矩阵
vectorizer = TfidfVectorizer(max_features=1000)
X_train_text = vectorizer.fit_transform(df_train['weibo_text'] + " " + df_train['tokenized_text']).toarray()

# 目标变量：将评分者1和评分者2的分数作为目标
y_avoidance_train = df_train[['opposite_sex_social_avoidance_1', 'opposite_sex_social_avoidance_2']]

# 对预测数据集进行特征提取
X_predict_text = vectorizer.transform(df_predict['weibo_text'] + " " + df_predict['tokenized_text']).toarray()

# 初始化随机森林回归器
# 使用 MultiOutputRegressor 将其扩展为多输出回归
rf_model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))

# 训练模型
rf_model.fit(X_train_text, y_avoidance_train)

# 对训练集进行预测
y_train_pred = rf_model.predict(X_train_text)

# 对新数据进行预测
y_predict = rf_model.predict(X_predict_text)

# 聚合预测结果（取评分者1和评分者2的预测值的平均值）
df_predict['predicted_opposite_sex_social_avoidance'] = y_predict.mean(axis=1)

# 保存预测结果到CSV文件，并使用 'utf-8-sig' 编码避免乱码
output_file = 'predicted_scores_rf_multi_output.csv'
df_predict.to_csv(output_file, encoding='utf-8-sig', index=False)

print(f'预测结果已保存到文件: {output_file}')
print(df_predict.head())  # 检查前几行结果，确保无误

# 计算模型的MSE和R²
# 由于是多输出回归，计算每个输出的 MSE 和 R²
mse_rf_1 = mean_squared_error(y_avoidance_train.iloc[:, 0], y_train_pred[:, 0])  # 对评分者1的误差
mse_rf_2 = mean_squared_error(y_avoidance_train.iloc[:, 1], y_train_pred[:, 1])  # 对评分者2的误差
r2_rf_1 = r2_score(y_avoidance_train.iloc[:, 0], y_train_pred[:, 0])
r2_rf_2 = r2_score(y_avoidance_train.iloc[:, 1], y_train_pred[:, 1])

print(f'MSE for Rater 1: {mse_rf_1}, MSE for Rater 2: {mse_rf_2}')
print(f'R² for Rater 1: {r2_rf_1}, R² for Rater 2: {r2_rf_2}')

# 计算RMSE
rmse_rf_1 = np.sqrt(mse_rf_1)
rmse_rf_2 = np.sqrt(mse_rf_2)
print(f'RMSE for Rater 1: {rmse_rf_1}, RMSE for Rater 2: {rmse_rf_2}')

# 计算MAE
mae_rf_1 = mean_absolute_error(y_avoidance_train.iloc[:, 0], y_train_pred[:, 0])
mae_rf_2 = mean_absolute_error(y_avoidance_train.iloc[:, 1], y_train_pred[:, 1])
print(f'MAE for Rater 1: {mae_rf_1}, MAE for Rater 2: {mae_rf_2}')




