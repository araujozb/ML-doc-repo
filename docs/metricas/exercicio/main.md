## Objetivo

Aplicar o algoritmo K-Nearest Neighbors (KNN) em um conjunto de dados de classificação, utilizando a mesma Aplicar os algoritmos K-Nearest Neighbors (KNN) e K-Means em conjuntos de dados distintos, com finalidades complementares:

Utilizar o Mushroom Dataset (via OpenML) para um problema de classificação binária (comestível vs venenoso), avaliando o desempenho do KNN com métricas de classificação tradicionais.

Utilizar o Mall Customers Dataset para um problema de segmentação de clientes, aplicando K-Means para identificar grupos com padrões de consumo semelhantes.

O objetivo geral é:
- Explorar, pré-processar e modelar conjuntos de dados reais;
- Avaliar o desempenho de um modelo supervisionado (KNN) e de um modelo não supervisionado (K-Means);
- Discutir diferenças de uso, interpretação e métrica entre classificação e clusterização.

## Escolha dos Datasets

### Mushroom Dataset (para o KNN, Classificação)
O Mushroom Dataset, obtido via OpenML (fetch_openml), descreve 8.124 cogumelos com 22 atributos categóricos (odor, cor das lamelas, textura do chapéu, tipo de anel, cor de esporos, habitat, etc.) e uma variável-alvo class com duas categorias:
-  → edible (comestível)
- p → poisonous (venenoso)

Esse dataset é adequado para classificação supervisionada, pois:
- Possui um rótulo claro (classe comestível/venenoso);
- É relativamente balanceado entre as classes;
- Já foi utilizado em exercícios anteriores (ex.: Árvore de Decisão), permitindo comparação entre algoritmos.

### Mall Customers Dataset (para K-Means, Segmentação)

O Mall Customers Dataset contém informações de 200 clientes de um shopping center, com as seguintes variáveis:
- CustomerID – identificador do cliente
- Gender – gênero (Male/Female)
- Age – idade
- Annual Income (k$) – renda anual (em milhares de dólares)
- Spending Score (1–100) – pontuação de gasto atribuída pelo shopping (comportamento de consumo)

Esse dataset é mais adequado para K-Means do que o Mushroom porque:
- As variáveis principais (idade, renda, score de gasto) são numéricas contínuas;
- K-Means depende da distância euclidiana, que funciona melhor nesse tipo de variável;
- O Mushroom é totalmente categórico e já é 100% separável pela classe, o que torna a clusterização menos interessante (os “grupos” seriam basicamente as próprias classes conhecidas).
---


## Parte 1 - KNN com o Mushroom (Classificação)

### Exploração dos dados (EDA)
Nesta etapa foi feita a análise exploratória básica do Mushroom Dataset, incluindo inspeção das primeiras linhas, colunas disponíveis e distribuição da variável-alvo.

=== "Code"
    ```python
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.datasets import fetch_openml

    # Carregamento do dataset via OpenML
    mush = fetch_openml("mushroom", version=1, as_frame=True)
    df = mush.frame

    print("Primeiras linhas do dataset:")
    print(df.head())

    print("\nColunas disponíveis:")
    print(df.columns.tolist())

    # Variável alvo e preditoras
    y = df["class"]
    X = df.drop("class", axis=1)

    # Distribuição da variável alvo
    print("\nDistribuição da variável alvo:")
    print(y.value_counts())

    # Gráfico de barras da classe
    y.value_counts().plot(kind="bar")
    plt.title("Distribuição da classe (Mushroom)")
    plt.xlabel("Classe (e = comestível, p = venenoso)")
    plt.ylabel("Quantidade")
    plt.tight_layout()
    plt.show()

    # Exemplo de frequência de um atributo relevante
    print("\nExemplo de contagem de valores de 'odor':")
    print(df["odor"].value_counts())
    ```

=== "Explicação"
- O dataset Mushroom possui 8.124 amostras e 22 variáveis categóricas.
- A variável alvo class possui duas categorias:
    - e = comestível
    - p = venenoso
- A distribuição das classes é relativamente balanceada (com leve predominância de cogumelos comestíveis).
- O atributo odor se mostra fortemente discriminativo, sugerindo que alguns odores aparecem quase exclusivamente em uma das classes — indício de alta separabilidade do problema.

--- 
### Pré-processamenro (Mushroom)
Como todas as variáveis preditoras são categóricas, foi necessário transformá-las em variáveis numéricas através de One-Hot Encoding. A variável-alvo foi codificada como binária (e → 0, p → 1).

Em seguida, aplicou-se padronização (StandardScaler) para uso em KNN (e também para o K-Means no outro dataset, por coerência de abordagem baseada em distância).

=== "Code"
    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # One-Hot Encoding das variáveis categóricas
    X_encoded = pd.get_dummies(X, drop_first=True)

    # Mapeamento da classe para binário: e -> 0, p -> 1
    y_encoded = y.map({"e": 0, "p": 1})

    print("\nShape após one-hot encoding:", X_encoded.shape)

    # Divisão treino/teste para KNN
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y_encoded,
        test_size=0.3,
        random_state=42,
        stratify=y_encoded
    )

    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)

    # Padronização
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Dataset completo escalado (também usado em outros testes, se necessário)
    X_full_scaled = scaler.fit_transform(X_encoded)
    ```

    ==="Explicação"
- One-Hot Encoding é necessário para transformar categorias em vetores binários, permitindo uso de distância euclidiana no KNN.
- A classe foi convertida para 0 (comestível) e 1 (venenoso), facilitando o cálculo das métricas.
- A divisão em 70% treino / 30% teste foi estratificada, garantindo a mesma proporção de classes em ambos os conjuntos.
- A padronização (média 0, desvio padrão 1) é fundamental para KNN, pois evita que variáveis com escala maior dominem a distância.

---

### Treinamento do Modelo (KNN)
O modelo KNN foi treinado com k = 5 vizinhos, utilizando os dados escalados.

=== "Code"
    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import (
        confusion_matrix,
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        classification_report
    )

    # Treinamento do modelo KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)

    # Predição no conjunto de teste
    y_pred_knn = knn.predict(X_test_scaled)

    print("\n===== RESULTADOS KNN =====")
    print("Acurácia:", accuracy_score(y_test, y_pred_knn))
    print("\nRelatório de classificação (KNN):")
    print(classification_report(y_test, y_pred_knn, target_names=["comestível", "venenoso"]))
    ```

=== "Explicação"
    - O KNN decide a classe de uma nova instância com base nos k vizinhos mais próximos no espaço de features.
    - Com k = 5, o modelo busca um equilíbrio entre variância e viés, evitando tanto o overfitting de k=1 quanto a suavização excessiva de k muito alto.
    - Devido à alta separabilidade do Mushroom Dataset, o modelo atinge quase desempenho perfeito (acurácia ≈ 0.99).


### Matriz de Confusão e Métricas (KNN)
Para avaliar o modelo de forma mais detalhada, foram calculadas a matriz de confusão e as métricas de acurácia, precisão, recall e F1-score.

=== "Code"
    ```python
    # Matriz de confusão
    cm_knn = confusion_matrix(y_test, y_pred_knn)
    print("Matriz de confusão (KNN):")
    print(cm_knn)

    # Métricas
    acc_knn = accuracy_score(y_test, y_pred_knn)
    prec_knn = precision_score(y_test, y_pred_knn)
    rec_knn = recall_score(y_test, y_pred_knn)
    f1_knn = f1_score(y_test, y_pred_knn)

    print(f"\nAcurácia={acc_knn:.4f}  Precisão={prec_knn:.4f}  Recall={rec_knn:.4f}  F1={f1_knn:.4f}")

    # Plot da matriz de confusão
    plt.figure(figsize=(4, 3))
    plt.imshow(cm_knn, interpolation='nearest')
    plt.title('Matriz de Confusão - KNN')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["comestível", "venenoso"], rotation=45)
    plt.yticks(tick_marks, ["comestível", "venenoso"])
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.tight_layout()
    # plt.savefig("docs/metricas/img/confusion_matrix_knn.png", dpi=150)
    plt.show()
    ```

    === "Explicação"
    - A matriz de confusão mostra quantos cogumelos foram corretamente classificados como comestíveis ou venenosos, e quantos foram confundidos entre as classes.
    - Os resultados obtidos indicam:
        - Acurácia ≈ 0.99
        - Precisão ≈ 0.99
        - Recall ≈ 0.99
        - F1-score ≈ 0.99
    - Isso significa que tanto falsos positivos quanto falsos negativos são extremamente raros, o que é crítico em um problema onde classificar erroneamente um cogumelo venenoso como comestível pode ter consequências graves.

---

## Parte 2 - K-Means com o Mall Customers (Segmentação)

### Exploração dos Dados (EDA)
O conjunto Mall Customers é composto por 200 clientes, com as variáveis:
- CustomerID — identificador (não informativo para clusterização)
- Gender — gênero
- Age — idade
- Annual Income (k$) — renda anual em milhares de dólares
- Spending Score (1–100) — score de gasto, atribuído pelo shopping

A EDA indicou:
- Ausência de valores ausentes;
- Idade concentrada entre ~20 e 50 anos;
- Renda variando aproximadamente entre 15 e 140 (k$);
- Spending Score bem espalhado, sugerindo múltiplos perfis de consumo;
- Gênero relativamente equilibrado entre masculino e feminino.

=== "Gráfico"
    ![Distribuição das variáveis numéricas](./img/kmeans_elbow_method.png)
    ![Distribuição por Gênero](./img/distribuicao_genero.png)  
    ![Distribuição por Gênero](./img/age_x_annual_income.png)  

=== "Explicação"
    - Não há relação linear forte entre idade, renda e Spending Score — clientes com renda alta podem ter gasto baixo e vice-versa.
    - Esse comportamento reforça a escolha de técnicas de agrupamento, como K-Means, para identificar segmentos latentes de clientes.

---

### 2. Pré-processamento  (Mall Customers)
- A coluna CustomerID foi removida por não carregar informação útil para clusterização.
- A variável categórica Gender foi transformada em numérica (0 = Male, 1 = Female).
- Foram selecionadas as seguintes features para o modelo:
    - Gender_num
    - Age
    - Annual Income (k$)
    - Spending Score (1–100)
- As variáveis foram padronizadas com StandardScaler para evitar que renda (escala maior) domine a distância euclidiana.

---

### 3. Divisão dos Dados  (Treino/Teste)
Embora o K-Means seja não supervisionado, foi feita uma divisão dos dados em:
- 80% treino
- 20% teste

O objetivo não é “avaliar acurácia de previsão”, mas sim verificar se os clusters obtidos são estáveis quando o modelo é aplicado a um subconjunto não usado na inicialização dos centróides.

---
### 4. Treinamento do Modelo K-Means
#### 4.1 Método do Cotovelo (Elbow Method)
O Método do Cotovelo foi aplicado calculando a inércia (Within-Cluster Sum of Squares – WCSS) para diferentes valores de k.

O gráfico indicou uma queda significativa da inércia entre k = 1 e k = 4, com reduções marginais a partir de k = 5, sugerindo que a região de interesse está entre 4 e 5 clusters.

=== "Gráfico"
    ![Elbow Method](./img/kmeans_elbow_method)

#### 4.2 Silhouette Score 
Para complementar a escolha de k, foi calculado o Silhouette Score para valores de k entre 2 e 7, em treino e teste.

| K     | Silhouette Treino | Silhouette Teste |
| ----- | ----------------- | ---------------- |
| 2     | 0.2467            | 0.2619           |
| 3     | 0.2466            | 0.2040           |
| 4     | 0.2951            | 0.2333           |
| 5     | 0.3103            | 0.2260           |
| 6     | 0.3262            | 0.2962           |
| **7** | **0.3668**        | **0.2998**       |


=== "Explicação"
- O Silhouette Score avalia:
     - **Coesão interna** (quão próximo cada ponto está do centro do seu cluster);
     - **Separação** (quão distante está dos outros clusters).
- Valores entre 0.2 e 0.4 são comuns em problemas reais de segmentação de clientes, onde perfis de consumo tendem a se sobrepor.
- O melhor resultado foi obtido com **k = 7**, tanto em treino quanto em teste, indicando uma segmentação mais granulada e coerente.

#### 4.3 Treinamento Final (k = 7)
Com base nos resultados do Elbow e do Silhouette Score, foi adotado K-Means com k = 7 como modelo final.
- Treinado com 80% dos dados padronizados
- Utilizado n_init = 10 para maior estabilidade na escolha dos centróides
- Cada cliente foi associado a um dos 7 clusters, permitindo análise de perfis
---

### 5. Interpretação dos Clusters (k = 7)
A análise de médias por cluster (idade, renda, Spending Score, gênero médio) permitiu a seguinte interpretação:
- Cluster 0 — Mulheres maduras conservadoras:
    - Exclusivamente feminino
    - Idade média acima de 50 anos
    - Renda intermediária
    - Baixo Spending Score  
        → Clientes mais estáveis, pouco engajados em consumo impulsivo.

- Cluster 1 — Homens maduros conservadores:
    - Público masculino com maior idade média (~56 anos)
    - Renda mediana
    - Baixo Spending Score
        → Perfil semelhante ao cluster 0, porém masculino.  

- Cluster 2 — Jovens de alta renda – Gastadores (VIP masculino):
    - Homens ~33 anos
    - Alta renda (~87k)
    - Spending Score elevado
        → Segmento premium, alta relevância para campanhas exclusivas.  

- Cluster 3 — Mulheres jovens engajadas:
    - Mulheres ~26 anos
    - Renda mais baixa
    - Spending Score acima da média
        → Consumidoras impulsivas, sensíveis a promoções e experiências.  

- Cluster 4 — Homens jovens gastadores:
    - Jovens de baixa renda (~40k)
    - Spending Score elevado
        → Perfil de alto consumo relativo à renda, semelhante ao cluster 3, porém masculino.  

- Cluster 5 — Mulheres de alta renda – Gastadoras (VIP feminino):
    - Renda alta (~86k)
    - Spending Score muito elevado
    - Idade média ~32 anos
        → Grupo VIP feminino, altamente valioso para estratégias de fidelização.

Cluster 6 — Alta renda, baixo engajamento:
    - Maior renda média (~92k)
    - Spending Score muito baixo
        → Clientes de alto potencial não explorado, prioritários para ações de engajamento.  

---

## Comparação entre KNN (Mushroom) e K-Means (Mall Customers)

Natureza do problema:
- KNN (Mushroom) → problema de classificação binária supervisionada (com rótulos).
- K-Means (Mall) → problema de segmentação não supervisionada (sem rótulos).

Tipo de avaliação:
- KNN → avaliado com matriz de confusão, acurácia, precisão, recall e F1-score.
- K-Means → avaliado com Silhouette Score e análise qualitativa dos clusters (perfis de clientes).

Interpretação dos resultados:
- No Mushroom, os atributos (especialmente odor) tornam a separação entre comestíveis e venenosos quase perfeita → KNN atinge desempenho ≈ 99%.
- No Mall Customers, os clusters representam perfis de consumo, não rótulos “certos ou errados”. O objetivo é entender grupos e não “acertar uma classe”.

Ponto chave:
- KNN responde à pergunta: “Dado um novo cogumelo, ele é comestível ou venenoso?”
- K-Means responde à pergunta: “Quais tipos de clientes existem nesse shopping e quais são seus perfis?” 

## Conclusão Geral

O KNN aplicado ao Mushroom Dataset demonstrou que modelos supervisionados podem atingir desempenho quase perfeito em problemas bem estruturados, desde que o pré-processamento (One-Hot + escala) seja adequado.

O K-Means aplicado ao Mall Customers Dataset mostrou o potencial de técnicas não supervisionadas para identificar segmentos de clientes e gerar insights de negócio, mesmo sem rótulos explícitos.

Em termos educacionais, o trabalho ilustra claramente a diferença entre:

Classificação supervisionada com métricas formais (KNN + matriz de confusão)

Clusterização e segmentação com foco em interpretação de grupos (K-Means + Silhouette + perfis)
