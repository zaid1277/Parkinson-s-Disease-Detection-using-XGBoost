import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
# You can download from: https://archive.ics.uci.edu/ml/datasets/parkinsons
try:
    df = pd.read_csv('parkinsons.data')
except FileNotFoundError:
    print("Error: parkinsons.data file not found!")
    print("Please ensure parkinsons.data is in the same directory as this script.")
    exit()

print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nDataset info:")
print(df.info())
print("\nClass distribution:")
print(df['status'].value_counts())

# Separate features and target
X = df.drop(['name', 'status'], axis=1)
y = df['status']

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train XGBoost model
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
    eval_metric='logloss'
)

print("\nTraining XGBoost model...")
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

# Evaluate the model
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print("\n" + "="*60)
print("MODEL PERFORMANCE")
print("="*60)
print(f"Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"Testing Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_pred_test, target_names=['Healthy', 'Parkinson\'s']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_test)
print("\nConfusion Matrix:")
print(cm)

# Visualizations
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Healthy', 'Parkinson\'s'],
            yticklabels=['Healthy', 'Parkinson\'s'],
            cbar=True, linewidths=1, linecolor='gray')
axes[0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Actual', fontsize=12)
axes[0].set_xlabel('Predicted', fontsize=12)

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False).head(10)

colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_importance)))
bars = axes[1].barh(range(len(feature_importance)), feature_importance['importance'], color=colors)
axes[1].set_yticks(range(len(feature_importance)))
axes[1].set_yticklabels(feature_importance['feature'])
axes[1].set_xlabel('Importance Score', fontsize=12)
axes[1].set_title('Top 10 Feature Importances', fontsize=14, fontweight='bold')
axes[1].invert_yaxis()
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('parkinsons_results.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved as 'parkinsons_results.png'")
plt.show()
