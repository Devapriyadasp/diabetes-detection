# Diabetes Detection using SVM and Decision Tree

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
df = pd.read_csv("diabetes.csv")

# Show dataset info
print("Dataset shape:", df.shape)
print(df.head(), "\n")

# Check for missing values
print("Missing values:\n", df.isnull().sum(), "\n")

# Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# Split data
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train models
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)

dt_model = DecisionTreeClassifier(random_state=42, max_depth=4)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

# Evaluate
print("SVM Accuracy:", accuracy_score(y_test, svm_pred))
print("Decision Tree Accuracy:", accuracy_score(y_test, dt_pred))
print("\nSVM Report:\n", classification_report(y_test, svm_pred))
print("\nDecision Tree Report:\n", classification_report(y_test, dt_pred))

# Confusion matrices
fig, ax = plt.subplots(1, 2, figsize=(10,4))
sns.heatmap(confusion_matrix(y_test, svm_pred), annot=True, fmt='d', cmap="Purples", ax=ax[0])
ax[0].set_title("SVM Confusion Matrix")
sns.heatmap(confusion_matrix(y_test, dt_pred), annot=True, fmt='d', cmap="Greens", ax=ax[1])
ax[1].set_title("Decision Tree Confusion Matrix")
plt.show()

# Accuracy comparison
models = ['SVM', 'Decision Tree']
accuracy = [accuracy_score(y_test, svm_pred), accuracy_score(y_test, dt_pred)]
plt.bar(models, accuracy, color=['purple','teal'])
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.show()

# Visualize decision tree
plt.figure(figsize=(15,8))
plot_tree(dt_model, filled=True, feature_names=df.columns[:-1], class_names=["No Diabetes","Diabetes"])
plt.title("Decision Tree Visualization")
plt.show()
