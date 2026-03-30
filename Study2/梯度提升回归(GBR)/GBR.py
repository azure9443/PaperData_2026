import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
import joblib


# 加载数据
df_train = pd.read_csv('part.csv', encoding='utf-8')
df_predict = pd.read_csv('full.csv', encoding='utf-8')

# 检查数据结构
print(df_train.head())
print(df_predict.head())


# 特征提取：使用文本数据创建TF-IDF矩阵
vectorizer = TfidfVectorizer(max_features=1000)
X_train_text = vectorizer.fit_transform(df_train['weibo_text'] + " " + df_train['tokenized_text']).toarray()

# 目标变量
y_avoidance_train = df_train['opposite_sex_social_avoidance']

# 训练GBM模型
gb_avoidance = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

# 拟合模型
gb_avoidance.fit(X_train_text, y_avoidance_train)

# 保存模型
joblib.dump(gb_avoidance, 'gb_avoidance_model.pkl')

# 对新的5万条数据进行预测
X_predict_text = vectorizer.transform(df_predict['weibo_text'] + " " + df_predict['tokenized_text']).toarray()
df_predict['Predicted_Social_avoidance'] = gb_avoidance.predict(X_predict_text)

# 保存预测结果到CSV文件
output_file = 'predicted_results.csv'
df_predict.to_csv(output_file, encoding='utf-8-sig', index=False)

print(f'预测结果已保存到文件: {output_file}')
print(df_predict.head())  # 检查前几行结果，确保无误

from sklearn.metrics import mean_squared_error, r2_score

# 使用训练集上的预测值来评估模型性能
y_avoidance_pred_train = gb_avoidance.predict(X_train_text)

# 计算MSE
mse_avoidance = mean_squared_error(y_avoidance_train, y_avoidance_pred_train)

# 计算R²
r2_avoidance = r2_score(y_avoidance_train, y_avoidance_pred_train)

print(f'MSE for Opposite-Sex Social avoidance Prediction: {mse_avoidance}')
print(f'R² for Opposite-Sex Social avoidance Prediction: {r2_avoidance}')


import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# 计算评估指标
mad = mean_absolute_error(y_avoidance_train, y_avoidance_pred_train)
rmse = np.sqrt(mean_squared_error(y_avoidance_train, y_avoidance_pred_train))
r2 = r2_score(y_avoidance_train, y_avoidance_pred_train)

# 计算 MAE (Mean Absolute Error)
mae_avoidance = mean_absolute_error(y_avoidance_train, y_avoidance_pred_train)
print(f'MAE for Opposite-Sex Social avoidance Prediction: {mae_avoidance}')


# 绘制实际值与预测值的对比图
plt.figure(figsize=(8, 6))
plt.scatter(y_avoidance_train, y_avoidance_pred_train, color='cyan')
plt.plot([y_avoidance_train.min(), y_avoidance_train.max()], [y_avoidance_train.min(), y_avoidance_train.max()], 'k--', lw=2)
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
plt.title('GBR Regression Result')
plt.text(0.05, 0.95, f'MAD={mad:.2f}\nR²={r2:.2f}\nRMSE={rmse:.2f}', transform=plt.gca().transAxes, 
         verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

# 保存图像
plt.savefig('gbr_regression_result_fixed.png', bbox_inches='tight', dpi=300)

# 显示图像
plt.show()

# 关闭图像窗口
plt.close()


交叉验证


from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

# 设定MSE作为评分标准
mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

# 执行10折交叉验证
cv_scores = cross_val_score(gb_avoidance, X_train_text, y_avoidance_train, cv=10, scoring=mse_scorer)

# 计算每次交叉验证的MSE值以及平均MSE
mean_cv_mse = -np.mean(cv_scores)
std_cv_mse = np.std(cv_scores)

print(f"10折交叉验证的平均MSE: {mean_cv_mse}")
print(f"10折交叉验证的MSE标准差: {std_cv_mse}")

import numpy as np

# 计算RMSE
rmse_avoidance = np.sqrt(mse_avoidance)

# 计算实际值的平均值
mean_actual_avoidance = np.mean(y_avoidance_train)

# 计算RRMSE
rrmse_avoidance = rmse_avoidance / mean_actual_avoidance

print(f'RRMSE for Opposite-Sex Social avoidance Prediction: {rrmse_avoidance}')


