import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
IMG_DIR = "docs/kmeans/exercicio/img/"
os.makedirs(IMG_DIR, exist_ok=True)

plt.rcParams['figure.figsize'] = (8, 5)
sns.set_style('whitegrid')

df = pd.read_csv("src/Mall_Customers.csv")


df.info("\n")
df.head()
df.describe()
df.isna().sum()

#distribuição das variáveis numéricas
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

sns.histplot(df["Age"], kde=True, ax=axes[0])
axes[0].set_title("Distribuição de Idade")

sns.histplot(df["Annual Income (k$)"], kde=True, ax=axes[1])
axes[1].set_title("Distribuição de Renda Anual (k$)")

sns.histplot(df["Spending Score (1-100)"], kde=True, ax=axes[2])
axes[2].set_title("Distribuição de Spending Score")

plt.tight_layout()
plt.show()
plt.savefig(os.path.join(IMG_DIR,"distribuicao_variaveis_numericas.png"))

#distribuição por genero
sns.countplot(data=df, x="Gender") 
plt.title("Distribuição por Gênero")
plt.show()
plt.savefig(os.path.join(IMG_DIR,"distribuicao_genero.png"))

#relações entre as variáveis (gráficos 2D)
#Idade x Renda
sns.scatterplot(
    data=df,
    x="Age",
    y="Annual Income (k$)"
)
plt.title("Age x Annual Income")
plt.show()
plt.savefig(os.path.join(IMG_DIR,"age_x_annual_income.png"))

#Renda x Spending Score
sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)"
)
plt.title("Annual Income x Spending Score")
plt.show()
plt.savefig(os.path.join(IMG_DIR,"annual_income_x_spending_score.png"))

#idade x spending score
sns.scatterplot(
    data=df,
    x="Age",
    y="Spending Score (1-100)"
)
plt.title("Age x Spending Score")
plt.show()
plt.savefig(os.path.join(IMG_DIR,"age_x_spending_score.png"))

sns.pairplot(
    df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]],
    diag_kind="kde"
)
plt.show()
plt.savefig(os.path.join(IMG_DIR,"pairplot_numericas.png"))

#Matriz de correlação
corr = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].corr()

sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues")
plt.title("Matriz de Correlação")
plt.show()
plt.savefig(os.path.join(IMG_DIR,"matriz_correlacao.png"))
