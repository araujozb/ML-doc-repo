from sklearn.datasets import fetch_openml #busca os datasets direto no site openML
import pandas as pd
import matplotlib.pyplot as plt



#caminho da pastas de imagens p/ o mkdocs ler
IMG_DIR = '.../decision-tree/img/'

#objeto bush com varios atributos
mush = fetch_openml(name="mushroom", version=1, as_frame=True) #as_frame retorna o dataset como dataframe do pd

#criando um df a partir do obj mush, pegando o atributo .frame
df = mush.frame


# visualizando os dados
print(df.head(20)) # limit 20 no df
print(df.info()) # dá um describe se é category ou não ;; binario aparece como category pq no fim é a mesma coisa
print("Shape:", df.shape) #qtd LxC
print(df.describe(include='all').T.head(20)) #gera estat. descritivas, funciona c/ col numericas mas o include=all // o T transpõe o df de colunas ---> linhas
print("\nDistribuição da classe:\n", df["class"].value_counts()) # conta quantas vezes cada valor da coluna class aparece [e/p]

# contagem de valores ausentes

col = df.columns
print("\nValores 'NaN' por coluna (potenciais ausentes):")
for col in df.columns:
    print(f"{col}: {(df[col] == 'NaN').sum()}")

# validando a contagem de cima com uma coluna especifica
print((df["stalk-root"] == "NaN").sum())

# gráfico de barra simples ---> column odor // Odor Distribution 
# 5.1 – Barras simples de 'odor'
df["odor"].value_counts().plot(kind="bar", title="Odor Distribution")
plt.tight_layout()
plt.savefig(IMG_DIR + "bar_odor.png")   # salva
plt.show()                              # mostra na tela
plt.clf()
df["odor"].value_counts().plot(kind="bar", title="Frequência de ODOR")
plt.tight_layout(); plt.savefig(IMG_DIR + "bar_odor.png"); plt.clf()
plt.show()