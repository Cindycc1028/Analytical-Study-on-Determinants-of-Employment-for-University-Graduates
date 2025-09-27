# Analytical-Study-on-Determinants-of-Employment-for-University-Graduates
该数据集模拟了10,000名大学生的学术与职业档案，聚焦于影响就业结果的关键因素。包含智商（IQ）、学业表现、累积平均绩点（CGPA）、实习经历、沟通能力等特征。
###SQL数据预处理
-- 1. 查看数据概览
SELECT 
    COUNT(*) as total_records,
    COUNT(DISTINCT College_ID) as unique_colleges
FROM college_student_placement_dataset;

-- 2. 检查缺失值
SELECT 
    SUM(CASE WHEN IQ IS NULL THEN 1 ELSE 0 END) as missing_IQ,
    SUM(CASE WHEN CGPA IS NULL THEN 1 ELSE 0 END) as missing_CGPA,
    SUM(CASE WHEN Placement IS NULL THEN 1 ELSE 0 END) as missing_Placement
FROM college_student_placement_dataset;

-- 3. 查看录用与未录用的人数
SELECT 
    Placement,
    COUNT(*) as count
FROM college_student_placement_dataset
GROUP BY Placement;

-- 4. 按学院统计录用率
SELECT 
    College_ID,
    AVG(CASE WHEN Placement = 'Yes' THEN 1 ELSE 0 END) * 100 as placement_rate
FROM college_student_placement_dataset
GROUP BY College_ID
ORDER BY placement_rate DESC;



###Python建模分析
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
