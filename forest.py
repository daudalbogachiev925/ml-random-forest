"""Random Forest с полным анализом."""
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np

# === 1. Данные ===
X, y = make_classification(n_samples=2000, n_features=20, n_informative=10,
                            random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 2. Сравнение: одно дерево vs лес ===
tree = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
forest = RandomForestClassifier(n_estimators=100, oob_score=True,
                                 random_state=42).fit(X_train, y_train)

print(f"Decision Tree: {accuracy_score(y_test, tree.predict(X_test)):.4f}")
print(f"Random Forest: {accuracy_score(y_test, forest.predict(X_test)):.4f}")
print(f"OOB Score:     {forest.oob_score_:.4f}")

# === 3. Подбор n_estimators ===
scores = []
for n in [10, 50, 100, 200, 300, 500]:
    rf = RandomForestClassifier(n_estimators=n, random_state=42)
    score = cross_val_score(rf, X_train, y_train, cv=3, scoring='roc_auc').mean()
    scores.append(score)
    print(f"n_estimators={n}: AUC={score:.4f}")

plt.plot([10, 50, 100, 200, 300, 500], scores, marker='o')
plt.xlabel("n_estimators"); plt.ylabel("ROC-AUC")
plt.title("Зависимость качества от числа деревьев")
plt.grid(True)
plt.savefig("n_estimators.png")
plt.show()

# === 4. Важность признаков ===
importances = forest.feature_importances_
top_idx = np.argsort(importances)[::-1][:10]
print("Топ-10 признаков:")
for i in top_idx:
    print(f"Feature {i}: {importances[i]:.4f}")
