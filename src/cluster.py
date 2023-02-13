# Importações de libraries importantes
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
import numpy as np
from sklearn.cluster import KMeans

# Carregamento do dataset
df = pd.read_csv("../data/dataset.csv")
del df["id"]

# Definição das features
X = df[["rat_weight", "rat_height", "rat_bodywater"]]
X = np.array(X)

# Teste para saber melhor número de clusters através do método elbow
K = range(1, 10)
distorcoes = []

for k in K:
    kmean = KMeans(n_clusters=k)
    kmean.fit(X)
    distorcoes.append(kmean.inertia_)
    
# Plota o método elbow (distoções vs no. de clusters)
plt.figure(figsize=(12,6))
plt.plot(K, distorcoes, 'bx-')
plt.xlabel('Clusters')
plt.ylabel('Distorções')
plt.title('Melhores números de clusters através do Método Elbow')
plt.show()

# Fita o modelo com o melhor número de clusters (n=3) identificado usando o gráfico do método elbow
kmean_model = KMeans(n_clusters=3, random_state=23994)
predict = kmean_model.fit_predict(X)
df["Predicted"] = predict


# Plota os clusters
plt.scatter(X[predict == 0, 0], X[predict == 0, 1], s = 100, c = 'red', label = 'Ratus norvegicus')
plt.scatter(X[predict == 1, 0], X[predict == 1, 1], s = 100, c = 'blue', label = 'Ratus terranus')
plt.scatter(X[predict == 2, 0], X[predict == 2, 1], s = 100, c = 'green', label = 'Ratus spatios')
# Plota os centroides dos clusters
plt.scatter(kmean_model.cluster_centers_[:, 0], kmean_model.cluster_centers_[:,1], s = 100, c = 'yellow', label = 'Centroides')
plt.legend()
plt.show()

# Plota a classificação real dos dados clusterizados
sbn.relplot(data=df, x=df["rat_height"], y=df["rat_weight"], hue=df["species"])
plt.show()