from sklearn.datasets import fetch_openml
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

mush = fetch_openml(name="mushroom", version=1, as_frame=True)
df = mush.frame

# nulls replacement
df.replace("?", np.nan, inplace=True)


#features x target
x = df.drop(columns=["class"]) #dropando a coluna class, pois todas as outras são features
y = df["class"] #coluna target



