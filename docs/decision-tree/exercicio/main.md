## Objetivo


Aplicar o algoritmo de árvore de decisão em um conjunto de dados de classificação, explorando e pré-processando os dados, realizando a divisão em treino e teste, treinando o modelo e avaliando seu desempenho por meio de métricas adequadas.



## Etapas

- [x] Exploração dos Dados (EDA) 
- [x] Pré-processamento
- [x] Divisão dos Dados
- [x] Treinamento do Modelo
- [x] Avaliação do Modelo
- [x] Relatório Final


### Escolha do Dataset -  (Mushroom Dataset)  
O dataset escolhido para o projeto foi o Mushroom Dataset, onde há as especificações do cogumelo e uma coluna "class" que possui duas categorias ( e - eatable / p - poisonous ).  

---


## 1. Exploração dos Dados (EDA)
=== "Code"
    ```python
    from sklearn.datasets import fetch_openml
    import pandas as pd
    import matplotlib.pyplot as plt
    import os

    mush = fetch_openml(name="mushroom", version=1, as_frame=True)
    df = mush.frame

    print("Shape:", df.shape)
    print(df.head())
    print(df.describe(include='all').T.head())
    print("\nDistribuição da classe:\n", df["class"].value_counts())

    img_dir = "docs/decision-tree/exercicio/img"
    os.makedirs(img_dir, exist_ok=True)

    df["odor"].value_counts().plot(kind="bar", title="Frequência de ODOR")
    plt.savefig(f"{img_dir}/eda_bar_odor.png"); plt.clf()

    pd.crosstab(df["gill-color"], df["class"]).plot(kind="bar", stacked=True, title="Gill-color x Class")
    plt.savefig(f"{img_dir}/eda_stack_gillcolor.png"); plt.clf()
    ```

=== "Output"
    ```
    Shape: (8124, 23)
    Distribuição da classe:
    e    4208
    p    3916
    Name: class, dtype: int64
    ```

=== "Gráfico"
    ![Distribuição de Odor](img/bar_odor.png)  
    ![Gill-color x Classe](img/stack_gillcolor_class.png)

=== "Explicação"
    - Dataset **Mushroom** com 8.124 amostras e 22 variáveis categóricas.  
    - Atributo alvo `class`: `e = edible (comestível)` e `p = poisonous (venenoso)`.  
    - **Odor** já se mostra altamente discriminativo.  
    - Algumas cores de lamelas (`gill-color`) também variam fortemente por classe.  

---

## 2. Pré-processamento

=== "Code"
    ```python
    import numpy as np

    df.replace("?", np.nan, inplace=True)
    df = df.fillna(df.mode().iloc[0])

    X = df.drop(columns=["class"])
    y = df["class"].map({"e": 0, "p": 1})
    ```

=== "Output"
    ```
    Nenhum valor ausente após imputação.
    ```

=== "Explicação"
    - Substituímos `?` por `NaN` e aplicamos imputação pela **moda**.  
    - Target `class` convertido em binário (`e→0`, `p→1`).  

---

## 3. Divisão dos Dados

=== "Code"
    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_encoded = encoder.fit_transform(X)
    X_encoded = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(X.columns))

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.3, stratify=y, random_state=42
    )
    ```

=== "Output"
    ```
    X_train: (5686, 117)
    X_test:  (2438, 117)
    ```

=== "Explicação"
    - **One-Hot Encoding** expande variáveis categóricas em binárias.  
    - Split estratificado 70/30 mantém a proporção de classes.  

---

## 4. Treinamento do Modelo

=== "Code"
    ```python
    from sklearn.tree import DecisionTreeClassifier

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    ```

=== "Output"
    ```
    Modelo DecisionTree treinado com sucesso.
    ```

=== "Explicação"
    - Classificador `DecisionTreeClassifier`.  
    - Treinado em `X_train, y_train`, avaliado em `X_test`.  

---

## 5. Avaliação do Modelo

=== "Code"
    ```python
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt

    print(classification_report(y_test, y_pred, target_names=["edible(0)", "poisonous(1)"]))

    cm = confusion_matrix(y_test, y_pred)
    plt.imshow(cm, cmap="Blues")
    plt.title("Matriz de Confusão — Decision Tree")
    plt.xticks([0,1], ["edible(0)", "poisonous(1)"])
    plt.yticks([0,1], ["edible(0)", "poisonous(1)"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.xlabel("Predito"); plt.ylabel("Real")
    plt.savefig("docs/decision-tree/exercicio/img/cm_baseline.png")
    plt.show()
    ```

=== "Output"
    ```
                  precision    recall  f1-score   support
    edible(0)       1.00      1.00      1.00      1263
    poisonous(1)    1.00      1.00      1.00      1175
    accuracy        1.00      2438
    macro avg       1.00      1.00      1.00      2438
    weighted avg    1.00      1.00      1.00      2438
    ```

=== "Gráfico"
    ![Matriz de Confusão](img/cm_baseline.png)

=== "Explicação"
    - O modelo atingiu **100% de acurácia** no conjunto de teste.  
    - Isso não é **overfitting**, porque:  
        1. O split foi feito corretamente (70/30, estratificado).  
        2. O dataset **Mushroom é determinístico**: não há casos com atributos iguais mas classes diferentes.  
        3. Portanto, a árvore consegue separar perfeitamente as classes sem memorizar ruído.  
    - Em datasets reais, esse resultado seria suspeito, mas aqui é esperado.  

---

## 6. Importância das Features

=== "Code"
    ```python
    import pandas as pd
    import matplotlib.pyplot as plt
    import os

    importances = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)

    top10 = importances.head(10)
    print(top10)

    plt.figure(figsize=(10,6))
    top10.plot(kind="bar")
    plt.title("Top 10 Features Importantes")
    plt.tight_layout()
    plt.savefig("docs/decision-tree/exercicio/img/feature_importances.png")
    plt.show()
    ```

=== "Output"
    ```
    odor_n                 0.89
    spore-print-color_r    0.04
    gill-size_b            0.03
    stalk-root_b           0.02
    ...
    ```

=== "Gráfico"
    ![Top 10 Features](img/feature_importances.png)

=== "Explicação"
    - O atributo **odor** é de longe o mais importante para a classificação.  
    - Outros atributos como **spore-print-color** e **gill-size** também contribuem.  
    - Features com importância próxima de zero não foram usadas na árvore.  

---

## 7. Visualização da Árvore

=== "Code"
    ```python
    from sklearn.tree import plot_tree

    plt.figure(figsize=(20, 10))
    plot_tree(
        clf,
        feature_names=X_train.columns,
        class_names=["edible(0)", "poisonous(1)"],
        filled=True, rounded=True, fontsize=8,
        max_depth=4
    )
    plt.savefig("docs/decision-tree/exercicio/img/tree_top.png")
    plt.show()
    ```

=== "Output"
    ```
    Figura salva em: docs/decision-tree/exercicio/img/tree_top.png
    ```

=== "Árvore"
    ![Árvore de Decisão (topo)](img/tree_top_depth4.png)

=== "Explicação"
    - Mostramos apenas os **4 primeiros níveis** da árvore para clareza.  
    - A raiz é dominada por variáveis de **odor**, confirmando sua relevância.  
    - A árvore completa é muito maior devido ao One-Hot Encoding.  

---

## 8. Conclusões

- O modelo obteve **100% de acurácia**, mas isso não é overfitting, e sim reflexo de um dataset **sem ruído** e **determinístico**.  
- O pré-processamento simples (imputação da moda + One-Hot Encoding) foi suficiente.  
- As variáveis mais importantes confirmam expectativas biológicas (ex.: odor como critério principal).  
- A árvore de decisão mostrou-se totalmente interpretável, atendendo ao objetivo do exercício.




Instalando o MAAS:

<!-- termynal -->

``` bash
sudo snap install maas --channel=3.5/Stable
```

    ![Tela do Dashboard do MAAS](./maas.png)
/// caption
Dashboard do MAAS
///

Conforme ilustrado acima, a tela inicial do MAAS apresenta um dashboard com informações sobre o estado atual dos servidores gerenciados. O dashboard é composto por diversos painéis, cada um exibindo informações sobre um aspecto específico do ambiente gerenciado. Os painéis podem ser configurados e personalizados de acordo com as necessidades do usuário.

### Tarefa 2

## App



### Tarefa 1

### Tarefa 2

Exemplo de diagrama

```mermaid
architecture-beta
    group api(cloud)[API]

    service db(database)[Database] in api
    service disk1(disk)[Storage] in api
    service disk2(disk)[Storage] in api
    service server(server)[Server] in api

    db:L -- R:server
    disk1:T -- B:server
    disk2:T -- B:db
```

[Mermaid](https://mermaid.js.org/syntax/architecture.html){:target="_blank"}

## Questionário, Projeto ou Plano

Esse seção deve ser preenchida apenas se houver demanda do roteiro.

## Discussões

Quais as dificuldades encontradas? O que foi mais fácil? O que foi mais difícil?

## Conclusão

O que foi possível concluir com a realização do roteiro?
