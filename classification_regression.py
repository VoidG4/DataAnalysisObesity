import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

df = pd.read_csv('processed_df.csv')

y = df["NObeyesdad"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X = df.drop("NObeyesdad", axis=1)

categorical_cols = ['Gender', 'family_history_with_overweight', 'FAVC',
                    'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

svm = SVC(probability=True, random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))
print("SVM Classification Report:\n", classification_report(y_test, y_pred_svm))

cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

cm_svm = confusion_matrix(y_test, y_pred_svm)
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Greens')
plt.title("Confusion Matrix - SVM")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

df["Height_m"] = df["Height"] / 100  
df["BMI"] = df["Weight"] / (df["Height_m"] ** 2)

y_bmi = df["BMI"]

X_bmi = df.drop(columns=["BMI", "Height_m", "Weight", "Height", "NObeyesdad"])

categorical_cols = ['Gender', 'family_history_with_overweight', 'FAVC',
                    'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']

X_bmi_encoded = pd.get_dummies(X_bmi, columns=categorical_cols, drop_first=True)

X_train_bmi, X_test_bmi, y_train_bmi, y_test_bmi = train_test_split(X_bmi_encoded, y_bmi, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_bmi_scaled = scaler.fit_transform(X_train_bmi)
X_test_bmi_scaled = scaler.transform(X_test_bmi)

model_custom = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train_bmi_scaled.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1) 
])

model_custom.compile(optimizer='adam', loss='mse', metrics=['mae'])
history_custom = model_custom.fit(X_train_bmi_scaled, y_train_bmi, epochs=50, validation_split=0.2, batch_size=32, verbose=0)

base_model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train_bmi_scaled.shape[1],)),
    layers.Dense(64, activation='relu')
])
base_model.trainable = False  

model_transfer = models.Sequential([
    base_model,
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

model_transfer.compile(optimizer='adam', loss='mse', metrics=['mae'])
history_transfer = model_transfer.fit(X_train_bmi_scaled, y_train_bmi, epochs=50, validation_split=0.2, batch_size=32, verbose=0)

pred_custom = model_custom.predict(X_test_bmi_scaled).flatten()
pred_transfer = model_transfer.predict(X_test_bmi_scaled).flatten()

def evaluate(y_true, y_pred, label="Model"):
    print(f"--- {label} ---")
    print("MAE:", mean_absolute_error(y_true, y_pred))
    print("MAPE:", mean_absolute_percentage_error(y_true, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))

evaluate(y_test_bmi, pred_custom, "Custom FeedForward")
evaluate(y_test_bmi, pred_transfer, "Transfer Learning")


plt.plot(history_custom.history["val_loss"], label="Custom val_loss")
plt.plot(history_transfer.history["val_loss"], label="Transfer val_loss")
plt.xlabel("Epochs")
plt.ylabel("Validation Loss (MSE)")
plt.legend()
plt.title("Validation Loss Comparison")
plt.show()
