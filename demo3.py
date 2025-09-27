import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# 1. 加载数据
df = pd.read_csv('college_student_placement_dataset.csv')

# 2. 数据概览
print(df.head())
print(df.info())
print(df.describe())

# 3. 数据清洗
# 将 Yes/No 转为 1/0
df['Internship_Experience'] = df['Internship_Experience'].map({'Yes': 1, 'No': 0})
df['Placement'] = df['Placement'].map({'Yes': 1, 'No': 0})

# 4. 探索性分析
# 录用率
placement_rate = df['Placement'].mean()
print(f"Overall Placement Rate: {placement_rate:.2%}")

# 特征与录用的关系
plt.figure(figsize=(10, 6))
sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png')  # 先保存图片
plt.close()
# 5. 特征与目标变量分布
plt.figure()
sns.boxplot(x='Placement', y='CGPA', data=df)
plt.title('CGPA vs Placement')
plt.savefig('CGPA_Vs_placement.png')  # 先保存图片
plt.close()
# 6. 建模
features = ['IQ', 'Prev_Sem_Result', 'CGPA', 'Academic_Performance', 
            'Internship_Experience', 'Extra_Curricular_Score', 
            'Communication_Skills', 'Projects_Completed']
X = df[features]
y = df['Placement']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 7. 特征重要性
importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(importance)

plt.figure()
sns.barplot(data=importance, x='importance', y='feature')
plt.title('Feature Importance')
plt.savefig('Feature_importance.png')  # 先保存图片
plt.close()
