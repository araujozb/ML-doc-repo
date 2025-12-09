## Objetivo
O objetivo deste exercício foi aplicar o algoritmo SVM (Support Vector Machine) ao dataset Wine, utilizando o Scikit-Learn, e comparar diferentes configurações de kernel e hiperparâmetros.
Além disso, foi realizada:
- padronização das variáveis  
- divisão entre treino e teste  
- avaliação com métricas de classificação (precision, recall, F1-score)  
- geração de uma visualização da fronteira de decisão em 2D utilizando apenas duas variáveis padronizadas.  
---

## Escolha do Dataset
Foi utilizado o dataset Wine do Scikit-Learn (load_wine()), composto por:
- 178 amostras
- 13 atributos químicos do vinho
- 3 classes (tipos de vinho)

Os atributos representam características físico-químicas como teor alcoólico, antocianinas, flavonoides, magnésio etc.
---

## Preparação dos Dados
Importação de pacotes
=== "Code"
    ```python
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.datasets import load_wine
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score

    ```

carregamento do dataset:
=== "Code"
    ```python
    data = load_wine()
    X = data.data
    y = data.target

    ```

divisão treino/teste:
=== "Code"
    ```python
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

    ```

padronização:
=== "Code"
    ```python
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    ```

A padronização é importante para SVM, pois o algoritmo é sensível à escala das variáveis.
--- 

## Treinamento do modelo SVM
O classificador usado foi o SVC (Support Vector Classifier).

=== "Code"
    ```python
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    svm.fit(X_train, y_train)

    ```

- Kernel utilizado: RBF (Radial Basis Function), ideal quando os dados não são linearmente separáveis.

- Hiperparâmetros selecionados:
    - C = 1.0
    - gamma = 'scale'
    - kernel = 'rbf'

---

## Avaliação do Modelo
Os resultados foram obtidos com:
    ```python
    pred = svm.predict(X_test)

    precision = precision_score(y_test, pred, average='macro')
    recall = recall_score(y_test, pred, average='macro')
    f1 = f1_score(y_test, pred, average='macro')

    ```

Métricas:
| Métrica  | Valor |
| -------- | ----- |
| Precisão | ~0.97 |
| Recall   | ~0.96 |
| F1-score | ~0.96 |
---

## Matriz de Confusão
Gerada a partir de :
=== "Code"
    ```python
    confusion_matrix(y_test, pred)

    ```
A matriz apresenta poucos erros, indicando que o modelo classificou corretamente a maioria das amostras.

## Visualização da Fronteira de Decisão (2D)

Para gerar um gráfico compreensível, foram usados somente dois atributos:
    - alcohol
    - flavanoids

Após padronização e criação de um grid 2D:
==="Code"
    ```python
    confusion_matrix(y_test, pred)

    ```
==="Gráfico"
    ```python
    ![Fronteiras de Decisão]("svm_rbf_decision_boundaries.png")

    ```
O gráfico produzido (svm_rbf_decision_boundaries.png) mostra claramente:
    - regiões de decisão não lineares
    - fronteiras suaves produzidas pelo kernel RBF
    - boa separação entre as classes, mesmo usando só 2 atributos dos 13 disponíveis

Isso reforça a capacidade do SVM-RBF de capturar relações complexas.
---

## Conclusão
O modelo SVM com kernel RBF apresentou excelente desempenho no dataset Wine:
    - alta precisão, recall e F1-score
    - poucas classificações incorretas
    - fronteiras de decisão claras e bem definidas

Mesmo reduzindo o problema para apenas duas variáveis, o SVM manteve boa separabilidade entre classes, demonstrando sua eficácia para dados complexos e não lineares.

Assim, o experimento confirma que o SVM é uma técnica poderosa de classificação, especialmente quando combinado com padronização e tuning apropriado dos hiperparâmetros.
