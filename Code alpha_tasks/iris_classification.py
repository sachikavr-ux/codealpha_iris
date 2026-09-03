import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv(r'C:\Users\sachi\Downloads\Iris.csv')
print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset Info:")
print(df.describe())

le = LabelEncoder()
df['Species_encoded'] = le.fit_transform(df['Species'])

X = df.drop(['Id', 'Species', 'Species_encoded'], axis=1)
y = df['Species_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='rbf'),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"\n{name}: {acc*100:.2f}%")

best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
best_acc = results[best_model_name]

print(f"\n{'='*50}")
print(f"BEST MODEL: {best_model_name} ({best_acc*100:.2f}%)")

y_pred_best = best_model.predict(X_test_scaled)
print(f"\nClassification Report ({best_model_name}):")
print(classification_report(y_test, y_pred_best, target_names=le.classes_))

cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f'Confusion Matrix - {best_model_name}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=100)
plt.close()

plt.figure(figsize=(10, 6))
plt.bar(results.keys(), results.values(), color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
plt.ylabel('Accuracy')
plt.title('Model Comparison')
plt.xticks(rotation=45)
plt.ylim([0.8, 1.05])
for i, (name, acc) in enumerate(results.items()):
    plt.text(i, acc + 0.01, f'{acc*100:.2f}%', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=100)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(df[df['Species']=='Iris-setosa']['SepalLengthCm'],
                df[df['Species']=='Iris-setosa']['PetalLengthCm'],
                label='Setosa', alpha=0.7)
axes[0].scatter(df[df['Species']=='Iris-versicolor']['SepalLengthCm'],
                df[df['Species']=='Iris-versicolor']['PetalLengthCm'],
                label='Versicolor', alpha=0.7)
axes[0].scatter(df[df['Species']=='Iris-virginica']['SepalLengthCm'],
                df[df['Species']=='Iris-virginica']['PetalLengthCm'],
                label='Virginica', alpha=0.7)
axes[0].set_xlabel('Sepal Length (cm)')
axes[0].set_ylabel('Petal Length (cm)')
axes[0].set_title('Sepal Length vs Petal Length')
axes[0].legend()

sns.boxplot(x='Species', y='PetalWidthCm', data=df, ax=axes[1])
axes[1].set_title('Petal Width Distribution by Species')
plt.tight_layout()
plt.savefig('iris_visualization.png', dpi=100)
plt.close()

print(f"\nFiles saved: confusion_matrix.png, model_comparison.png, iris_visualization.png")
