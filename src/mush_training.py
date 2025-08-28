import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

data_dir = "data"
img_dir = "docs/decision-tree/exercicio/img"

os.makedirs(img_dir, exist_ok=True)

#carregar os dados já pré-processados/CSVs
x_train = pd.read_csv(f"{data_dir}/dataset-x-train.csv")
x_test  = pd.read_csv(f"{data_dir}/dataset-x-test.csv")
y_train = pd.read_csv(f"{data_dir}/dataset-y-train.csv")["class"]
y_test  = pd.read_csv(f"{data_dir}/dataset-y-test.csv")["class"]

#modelo baseline
clf = DecisionTreeClassifier(random_state=42)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

# métricas

acc = accuracy_score(y_test, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(
    y_test, y_pred, average="macro", zero_division=0
)

print("\n== Classification Report (Baseline) ==")
print(classification_report(y_test, y_pred, target_names=["edible(0)", "poisonous(1)"], zero_division=0))
print(f"Accuracy: {acc:.4f} | Precision_macro: {prec:.4f} | Recall_macro: {rec:.4f} | F1_macro: {f1:.4f}")
