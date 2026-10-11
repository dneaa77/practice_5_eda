
# ПРАКТИЧЕСКАЯ РАБОТА №5
# Подбор порога классификации с учетом цены ошибок

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)

# 1. Загружаем датасет
data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)

# Делаем класс 1 = злокачественная опухоль
# Класс 0 = доброкачественная опухоль
y = (data.target == 0).astype(int)

print("Размер датасета:", X.shape)
print("Количество классов:")
print(y.value_counts().sort_index())

# 2. Разделяем данные на обучение, проверку и тест
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print("\nРазмер обучающей выборки:", len(X_train))
print("Размер проверочной выборки:", len(X_val))
print("Размер тестовой выборки:", len(X_test))

# 3. Создаем и обучаем модель
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=5000, random_state=42)
)

model.fit(X_train, y_train)

# 4. Получаем вероятности класса 1
# Берем именно вероятность злокачественной опухоли
positive_index = list(model.classes_).index(1)

prob_val = model.predict_proba(X_val)[:, positive_index]
prob_test = model.predict_proba(X_test)[:, positive_index]

print("\nПервые 10 вероятностей:")
print(np.round(prob_test[:10], 3))

# 5. Задаем цену ошибок
COST_FP = 1
COST_FN = 5

# Подбираем порог только на проверочной выборке
thresholds_to_check = np.arange(0.05, 0.96, 0.01)

best_threshold = 0.5
best_cost = float("inf")

for threshold in thresholds_to_check:
    pred_val = (prob_val >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val, pred_val, labels=[0, 1]
    ).ravel()

    total_cost = COST_FP * fp + COST_FN * fn

    if total_cost < best_cost:
        best_cost = total_cost
        best_threshold = threshold

print("\nЛучший порог:", round(best_threshold, 2))
print("Минимальная цена ошибок на проверочной выборке:", best_cost)

# 6. Сравниваем порог 0.5 и подобранный порог
def show_metrics(y_true, probabilities, threshold, title):
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    cost = COST_FP * fp + COST_FN * fn

    print(f"\n--- {title} ---")
    print("Порог:", round(threshold, 2))
    print("Accuracy:", round(accuracy_score(y_true, predictions), 3))
    print("Precision:", round(
        precision_score(y_true, predictions, zero_division=0), 3
    ))
    print("Recall:", round(recall_score(y_true, predictions), 3))
    print("F1:", round(f1_score(y_true, predictions, zero_division=0), 3))
    print("TN:", tn, "| FP:", fp, "| FN:", fn, "| TP:", tp)
    print("Общая цена ошибок:", cost)

    return predictions

pred_default = show_metrics(
    y_test, prob_test, 0.5, "Порог по умолчанию"
)

pred_tuned = show_metrics(
    y_test, prob_test, best_threshold, "Подобранный порог"
)

# 7. ROC-кривая и ROC-AUC
fpr, tpr, roc_thresholds = roc_curve(y_test, prob_test)
roc_auc = roc_auc_score(y_test, prob_test)

# 8. PR-кривая и Average Precision
precision_values, recall_values, pr_thresholds = (
    precision_recall_curve(y_test, prob_test)
)

pr_auc = average_precision_score(y_test, prob_test)

print("\nROC-AUC:", round(roc_auc, 3))
print("PR-AUC (Average Precision):", round(pr_auc, 3))

# 9. Рисуем два графика
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.3f}")
axes[0].plot([0, 1], [0, 1], linestyle="--")
axes[0].set_title("ROC-кривая")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(
    recall_values, precision_values,
    label=f"Average Precision = {pr_auc:.3f}"
)
axes[1].set_title("PR-кривая")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

print("\nРабота завершена!")
