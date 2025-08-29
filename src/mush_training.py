import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

#definição dos diretórios
data_dir = "data"
img_dir = "docs/decision-tree/exercicio/img"

os.makedirs(img_dir, exist_ok=True)

#carregar os dados já pré-processados/CSVs
#features 
x_train = pd.read_csv(f"{data_dir}/dataset-x-train.csv")
x_test  = pd.read_csv(f"{data_dir}/dataset-x-test.csv")
# target
y_train = pd.read_csv(f"{data_dir}/dataset-y-train.csv")["class"]
y_test  = pd.read_csv(f"{data_dir}/dataset-y-test.csv")["class"]

#modelo baseline
clf = DecisionTreeClassifier(random_state=42) #criando um classificador
clf.fit(x_train, y_train) #treina com os dados do treino
y_pred = clf.predict(x_test) #gera previsões no cj de testes

# métricas

acc = accuracy_score(y_test, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(
    y_test, y_pred, average="macro", zero_division=0
)


# AVALIAÇÃO DO MODELO
print("\n== Classification Report (Baseline) ==")
print(classification_report(y_test, y_pred, target_names=["edible(0)", "poisonous(1)"], zero_division=0))
print(f"Accuracy: {acc:.4f} | Precision_macro: {prec:.4f} | Recall_macro: {rec:.4f} | F1_macro: {f1:.4f}")

# matriz de confusão
labels = [0,1]
cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.imshow(cm)
plt.title("Matriz de confusão")
plt.xticks([0, 1], ["edible(0)", "poisonous(1)"])
plt.yticks([0, 1], ["edible(0)", "poisonous(1)"])
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(img_dir, "cm_baseline.png"))
plt.show()
plt.clf() 


# profundidade e numero de folhas
print(f"\nProfundidade de árvore: {clf.get_depth()}")
print(f"Número de folhas: {clf.get_n_leaves()}")


# feature_importances mede a importancia das features de acordo
# com quanto cada uma contribui para reduzir impurezassssssss
importances = pd.Series(clf.feature_importances_, index=x_train.columns).sort_values(ascending=False)
plt.figure(figsize=(10,6))
importances.head(10).plot(kind="bar")
plt.title("Top 10 Features Importantes")
plt.tight_layout()
plt.savefig("docs/decision-tree/exercicio/img/feature_importances.png")
plt.show()

