## Objetivo

Aplicar o algoritmo **K-Means** em um conjunto de dados para realizar **segmentação de clientes**.
Para tanto, foi alterado o dataset original (Mushroom), utilizando então o 
**Mall Customers Dataset (Customer Segmentation)**.

## Por que o Dataset foi alterado?

O **Mushroom Dataset** é totalmente categórico, todas as 22 features são categóricas e não numéricas.

O K-Means não lida bem com variáveis categóricas, por que ele usa distância euclidiana, que por sua vez, só funciona bem com variáveis contínuas.

Outro ponto é que o **Mush** é 100% separável pela classe (eatible x poisonous), isso diz que o dataset já possui uma separação perfeira, então segmentar com K-Means não encontra novos grupos.



## Etapas

- [x] Exploração dos Dados (EDA) 
- [x] Pré-processamento
- [x] Divisão dos Dados
- [x] Treinamento do Modelo
- [x] Avaliação do Modelo
- [x] Relatório Final


### 1. Exploração dos Dados (EDA)
O conjunto de dados Mall Customers é composto por 200 clientes de um shopping center, com as seguintes variáveis: CustomerID (identificador), Gender (gênero), Age (idade), Annual Income ($) (renda anual em milhares de dólares) e Spending Score (1–100) (pontuação atribuída pelo shopping com base no comportamento de compra).

A análise exploratória inicial mostrou que não há valores ausentes nas variáveis. As distribuições indicam que a idade dos clientes está concentrada entre aproximadamente 20 e 50 anos, enquanto a renda anual varia em torno de 15 a 140 mil dólares. O Spending Score apresenta uma distribuição bastante espalhada, o que sugere perfis de consumo bem diferentes dentro da base. A variável Gender está relativamente equilibrada entre clientes do sexo masculino e feminino.

Os gráficos de dispersão entre idade, renda e Spending Score indicam que não existe uma relação linear forte entre essas variáveis, o que reforça a necessidade de usar técnicas de agrupamento para identificar segmentos de clientes com padrões de comportamento semelhantes.

=== "Gráfico"
    ![Distribuição das variáveis numéricas](img/distribuicao_variaveis_numericas.png)  
    ![Distribuição por Gênero](img/distribuicao_genero.png)
    ![Idade x Renda](img/age_x_annual_income.png.png)

---

### 2. Pré-processamento  

O dataset Mall Customers apresentou-se inicialmente sem valores ausentes, garantindo boa qualidade dos dados para processamento. Como primeira etapa, a coluna CustomerID foi removida por se tratar apenas de um identificador e não conter informação relevante para o processo de clusterização.

A variável categórica Gender foi convertida em formato numérico, atribuindo-se o valor 0 para Male e 1 para Female. Essa transformação é necessária, pois o algoritmo K-Means opera exclusivamente com variáveis numéricas.

As features selecionadas para o modelo foram:

- *Gender_num*
- *Age*
- *Annual Income (k$)*
- *Spending Score (1–100)*

Por fim, todas as variáveis foram padronizadas utilizando o método *StandardScaler*, de forma que cada feature apresentasse média 0 e desvio padrão 1. Essa normalização é fundamental, uma vez que o K-Means utiliza distância Euclidiana e seria fortemente influenciado por variáveis com escalas maiores, como a renda anual.

---

### 3. Divisão dos Dados  

Embora o K-Means seja um algoritmo de aprendizado **não supervisionado**, a rubrica da atividade exige a separação dos dados em treino e teste.  

Assim, a matriz de features padronizadas foi dividida em:

- **80% para treino**  
- **20% para teste**

O objetivo dessa etapa é verificar a **estabilidade dos clusters** quando aplicados a um subconjunto de dados não utilizado na fase inicial de ajuste.

Essa divisão não afeta a construção dos clusters, mas auxilia na avaliação da consistência do modelo.

---

### 4. Treinamento do Modelo

### 4.1 Método do Cotovelo

Para determinar o número ideal de clusters, aplicou-se o método do cotovelo,
que consiste em calcular a inércia (WCSS – Within Cluster Sum of Squares)
para diferentes valores de *k*.

O gráfico obtido mostra uma queda acentuada da inércia entre **k = 1** e **k = 4**.
A partir de **k = 5**, a redução passa a ser marginal, caracterizando o ponto de 
diminuição dos ganhos, conhecido como “cotovelo”.

Com base nessa análise, o número de clusters mais apropriado para este dataset
situa-se entre **4 e 5**. Nos passos seguintes, utilizaremos **k = 5**, pois esse
valor costuma gerar segmentos mais úteis e facilmente interpretáveis.


=== "Gráfico"
    ![Distribuição das variáveis numéricas](img/kmeans_elbow_method.png)  

---


### 4.2 Interpretação do Silhouette Score

Após o ajuste do modelo K-Means para diferentes valores de *k* (entre 2 e 7), 
foi calculado o **Silhouette Score**, métrica que avalia simultaneamente:

- **coesão interna**, ou seja, quão próximos os pontos estão do seu próprio cluster;  
- **separação**, isto é, quão distantes estão dos clusters vizinhos.

O Silhouette varia entre -1 e 1. Valores próximos de **1** indicam clusters bem 
separados; valores próximos de **0** indicam sobreposição entre grupos; e valores 
negativos sugerem agrupamentos inadequados.

Os resultados obtidos foram:
| K | Silhouette Treino | Silhouette Teste |
|---|-------------------|------------------|
| 2 | 0.2467 | 0.2619 |
| 3 | 0.2466 | 0.2040 |
| 4 | 0.2951 | 0.2333 |
| 5 | 0.3103 | 0.2260 |
| 6 | 0.3262 | 0.2962 |
| **7** | **0.3668** | **0.2998** |

Embora esses valores não sejam elevados (idealmente entre 0.4 e 0.5), eles se 
enquadram dentro do esperado para problemas reais de segmentação, nos quais 
os perfis dos clientes tendem a apresentar **transições suaves** e não 
fronteiras rígidas. Em bases de dados comportamentais, é comum que diferentes 
grupos de consumidores possuam características parcialmente sobrepostas, o que 
naturalmente reduz o Silhouette Score.

Entre todos os valores testados, o maior Silhouette foi obtido com **k = 7**, 
tanto no conjunto de treino quanto no teste. Isso indica que esse valor oferece 
o melhor equilíbrio entre coesão e separação dos clusters, representando o 
número de segmentos que mais adequadamente descreve a estrutura presente nos 
dados.

Assim, o modelo final adotado foi o **K-Means com 7 clusters**, refletindo uma 
segmentação mais granular e informativa dos perfis de clientes.




### 4.3 Treinamento Final do Modelo K-Means

Embora o Método do Cotovelo tenha sugerido um intervalo possível entre **k = 4**
e **k = 5**, a validação por meio do **Silhouette Score** demonstrou que o valor
de **k = 7** apresenta a melhor combinação entre coesão interna e separação entre
os clusters.

Assim, o modelo final foi treinado com **7 clusters**, utilizando 80% dos dados
(normalizados) e o parâmetro `n_init=10`, garantindo maior estabilidade na
inicialização dos centróides.

Após o ajuste, cada cliente foi atribuído a um dos sete clusters, tanto no
conjunto de treino quanto no teste, permitindo analisar a consistência e o
perfil dos segmentos identificados.

### 5. Interpretação dos Clusters (k = 7)

Após o treinamento final do modelo K-Means com **k = 7**, foi realizada uma
análise detalhada das características médias de cada grupo. A tabela de perfis
permite compreender o comportamento dos segmentos com base em quatro variáveis:
gênero, idade, renda anual e *spending score*.

A seguir, apresenta-se a interpretação dos clusters:

- **Cluster 0 — Mulheres maduras conservadoras:** composto exclusivamente por
  mulheres com idade média superior a 50 anos, renda intermediária e gasto
  baixo. Representa clientes de perfil mais estável e pouco engajado.

- **Cluster 1 — Homens maduros conservadores:** grupo masculino de maior idade
  média (56 anos), renda mediana e baixo consumo. Similar ao cluster 0, porém
  no público masculino.

- **Cluster 2 — Jovens de alta renda – Gastadores (VIP masculino):** formado por
  homens de cerca de 33 anos, com alta renda (≈87k) e elevado spending score.
  Segmento premium e altamente valioso.

- **Cluster 3 — Mulheres jovens engajadas:** mulheres por volta de 26 anos com
  renda mais baixa, porém gasto acima da média. Perfil impulsivo, responde bem
  a promoções.

- **Cluster 4 — Homens jovens gastadores:** jovens de baixa renda (≈40k), mas
  com spending score elevado. Segmento com comportamento de compra semelhante
  ao cluster 3, porém masculino.

- **Cluster 5 — Mulheres de alta renda – Gastadoras (VIP feminino):** renda alta
  (≈86k), spending score muito elevado e idade média de 32 anos. Grupo de grande
  valor comercial, ideal para ações premium e fidelização.

- **Cluster 6 — Alta renda, baixo engajamento:** grupo misto em gênero, com a
  maior renda média entre todos os clusters (≈92k), porém com gasto muito reduzido.
  Indica clientes de alto potencial não aproveitado.

Esses segmentos oferecem uma visão granular e estratégica do comportamento dos
clientes, permitindo ações direcionadas de marketing, retenção e fidelização.


### 4. Treinamento do Modelo  

O modelo K-Nearest Neighbors (KNN) foi treinado com diferentes valores de k (1, 3, 5, 7, 9, 11), buscando identificar o número ideal de vizinhos que maximiza o desempenho.

Como o algoritmo é baseado em distâncias, o One-Hot Encoding aplicado anteriormente garante que as categorias sejam interpretadas de forma binária e equidistante, evitando distorções no cálculo da similaridade entre amostras.

---

### 5. Avaliação do Modelo  

Foram calculadas métricas de acurácia, precisão, recall e F1-score, além da matriz de confusão.

Como no modelo anterior (árvore de decisão), o KNN também atingiu desempenho máximo — o que reforça a natureza determinística do dataset Mushroom. 


### 6. Conclusão

O objetivo deste trabalho foi aplicar o algoritmo **K-Means** em um problema de 
**segmentação de clientes**, utilizando o *Mall Customers Dataset* em substituição
ao conjunto de dados original (Mushroom). Essa alteração permitiu trabalhar com 
variáveis numéricas contínuas, mais adequadas ao uso da distância euclidiana, que 
é a base de funcionamento do K-Means.

A partir da análise exploratória, foi possível observar que idade, renda anual e 
*Spending Score* apresentam grande variabilidade e não exibem relações lineares 
fortes entre si. Isso reforça a escolha do K-Means como técnica de agrupamento,
já que o objetivo não é prever um rótulo, mas sim identificar **padrões de 
comportamento** em uma base heterogênea de clientes. O pré-processamento incluiu 
a remoção do identificador (*CustomerID*), a codificação da variável categórica 
(*Gender*) e a padronização das features, garantindo que todas contribuíssem de 
forma equilibrada para o cálculo das distâncias.

Na etapa de modelagem, o Método do Cotovelo foi utilizado para sugerir um intervalo 
inicial plausível para o número de clusters, enquanto o **Silhouette Score** foi 
empregado como critério quantitativo para comparar diferentes valores de *k*. Os 
melhores resultados foram obtidos com **k = 7**, que apresentou os maiores valores 
de Silhouette tanto em treino (0,3668) quanto em teste (0,2998). Embora esses 
valores não sejam elevados, eles são compatíveis com problemas reais de 
segmentação, nos quais os grupos tendem a se sobrepor parcialmente.

A interpretação dos sete clusters revelou perfis de clientes claramente distintos, 
como grupos de **alta renda e alto gasto (segmentos VIP)**, **jovens de baixa renda 
com alto Spending Score** e um segmento de **alta renda com baixo engajamento**, 
que representa potencial de crescimento ainda não explorado. Esses insights podem 
ser utilizados para orientar ações de marketing mais direcionadas, programas de 
fidelização e estratégias de retenção de clientes.

Como trabalhos futuros, seria interessante:
- incorporar novas variáveis comportamentais (frequência de visitas, categorias de produtos, canal de compra, etc.);
- comparar o desempenho do K-Means com outros algoritmos de clusterização 
  (como DBSCAN, GMM ou métodos hierárquicos);
- analisar a evolução dos clusters ao longo do tempo, avaliando mudanças no 
  comportamento dos segmentos identificados.

Em síntese, o uso do K-Means sobre o *Mall Customers Dataset* mostrou-se adequado 
para o problema proposto, permitindo identificar segmentos relevantes de clientes 
e demonstrando, na prática, a aplicação de técnicas de **aprendizado não 
supervisionado** em um contexto real de negócios.



