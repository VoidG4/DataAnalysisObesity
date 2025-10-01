import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv("data/obesity_dataset.csv")

df.head()

print(df.shape)
print(df.dtypes)

print(df.isnull().sum())

num_features = ['Age', 'Height', 'Weight']

plt.figure(figsize=(10, 6))
sns.boxplot(data=df[num_features])
plt.title("Boxplot αριθμητικών χαρακτηριστικών")
plt.show()

df = df.dropna() 

df = df[(df['Height'] > 1.2) & (df['Height'] < 2.2)]

scaler = StandardScaler()
df[['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']] = scaler.fit_transform(
    df[['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']]
)

categorical_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS', 'NObeyesdad']

df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
df_encoded.head()

plt.figure(figsize=(14, 10))
sns.heatmap(df_encoded.corr(), cmap='coolwarm', annot=False)
plt.title("Χάρτης συσχέτισης χαρακτηριστικών")
plt.show()

X = df_encoded.drop(columns=[col for col in df_encoded.columns if 'NObeyesdad' in col])
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1])
plt.title("PCA Visualization (2D)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()

sns.histplot(df['Age'], kde=True)
plt.title("Κατανομή ηλικιών")
plt.show()

sns.countplot(x='Gender', data=df)
plt.title("Κατανομή φύλου")
plt.show()

sns.countplot(x='NObeyesdad', data=df)
plt.title("Κατηγορίες Παχυσαρκίας")
plt.xticks(rotation=45)
plt.show()

df.to_csv('processed_df.csv', index=False)