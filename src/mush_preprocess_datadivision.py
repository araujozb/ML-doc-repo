from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# pré-processamento e divisao de dados 

mush = fetch_openml(name="mushroom", version=1, as_frame=True)
df = mush.frame

# nulls replacement
# replacing ? with NaN
df.replace("?", np.nan, inplace=True) # o inplace muda o df original, sem criar uma cópia
print("\nValores ausentes por coluna:\n", df.isna().sum())

#substitui os valores NaN pela moda
df = df.fillna(df.mode().iloc[0])
print("\nValores ausentes por coluna:\n", df.isna().sum()) #confere se ainda há valores ausentes


#features x target
x = df.drop(columns=["class"]) #dropando a coluna class, pois todas as outras são features
y = df["class"].map({"e": 0, "p": 1}) #coluna target [edible: 0, poisonous: 1]


# encoding
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
x_encoded_array = encoder.fit_transform(x)
encoded_feature_names = encoder.get_feature_names_out(x.columns)

#final df encoded
x = pd.DataFrame(x_encoded_array, columns=encoded_feature_names)


# train test / divisão estratificada

#train_test_split divide o dataset em dois subconjuntos (train [treinamento] e test [teste])
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.3, stratify=y, random_state=42 )

#salvando os datasets
os.makedirs("data", exist_ok=True)
# esse index=False é só para que o indice do dataset não seja uma columa a mais
x_train.to_csv("data/dataset-X-train.csv", index=False) 
x_test.to_csv("data/dataset-X-test.csv", index=False)
y_train.to_frame(name="class").to_csv("data/dataset-y-train.csv", index=False)
y_test.to_frame(name="class").to_csv("data/dataset-y-test.csv", index=False)

# nome das colunas codificadas
pd.Series(encoded_feature_names, name = 'feature').to_csv(
    "data/encoded_features.csv", index=False
)




