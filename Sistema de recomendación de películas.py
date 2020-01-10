#!/usr/bin/env python
# coding: utf-8

# # Sistema de recomendación de películas
# 

# In[34]:


#Librerias
import pandas as pd
import numpy as np
import nltk
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from scipy import stats
from scipy import sparse
import seaborn as sns
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from sklearn.preprocessing import MinMaxScaler
from surprise import Reader
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
from surprise.model_selection import GridSearchCV
from surprise import SVD, SVDpp, NMF, SlopeOne, CoClustering, KNNBaseline, KNNWithZScore, KNNWithMeans, KNNBasic, BaselineOnly, NormalPredictor
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity


# In[3]:


df_credits = pd.read_csv("credits.csv", sep = ',', low_memory=False)
df_movies = pd.read_csv("movies_metadata.csv", sep = ',', low_memory=False)
df_ratings = pd.read_csv("ratings_small.csv", sep = ',', low_memory=False)


# In[55]:


df_movies.shape


# In[4]:


df_ratings.head(2)


# In[5]:


df_credits.head(2)


# In[39]:


df_movies.describe()


# In[40]:


df_credits.describe()


# In[41]:


df_ratings.describe()


# Para trabajar en estos datos en combinación necesitamos concatenarlos para su mejor uso

# In[6]:


pd.concat([df_movies,df_credits == "id"], sort=False)
df_movies.head(3)


# # Sacamos la información de las peliculas que contiene la valoración por usuario

# In[35]:


n_users = df_ratings.userId.unique().shape[0]
n_movies = df_ratings.movieId.unique().shape[0]
print (str(n_users) + ' users')
print (str(n_movies) + ' movies')


# Tenemos 671 usuarios y 9066 peliculas valoradas

# In[36]:


plt.style.use('dark_background')
df_ratings['rating'].value_counts().head(80).plot.bar(color = 'orange', figsize = (20, 7))
plt.title('Valoraciones por usuarios', fontsize = 30, fontweight = 20)
plt.xlabel('Valoración')
plt.ylabel('count')
plt.show()


# Tenemos unos 28000 de valoraciones con una puntuación de 4 y unos 20000 con puntuación en 3. Veamos las cantidades exactas:

# In[37]:


df_ratings.groupby(["rating"])["userId"].count()


# # Filtrado demografico
# 

# Para realizar este filtrado utilizamos las calificaciones de las peliculas. Utilizaremos las calificaciones ponderadas:  Promedio ponderado (WR): - ((v / v + m) * R) + ((m / m + v) * C)
# Donde v es número de votos para la película, m es votos mínimos requeridos para ser listados en la tabla, R es rating promedio de la película, C es voto medio en todo el informe 

# In[43]:


#Calculamos C
c=df_movies['vote_average'].mean()
print(c)


# Filtramos las mejores peliculas, consideras aquellas con calificaciones superiores al percentil 90. Utilizamos como punto de referencia m

# In[46]:


#Calculamos m
m=df_movies['vote_count'].quantile(0.9)
m


# Seleccionamos las peliculas con calificaciones superiores

# In[9]:


qualified_movies=df_movies.copy().loc[df_movies['vote_count']>=m]
qualified_movies.shape


# Son 4555 películas seleccionadas; ahora daremos una calificación a cada película. Se define la función, imdbscore(), y una nueva puntuación de características, de la cual calcularemos el valor por aplicando esta función a nuestro DataFrame de películas calificadas

# In[10]:


def imdbscore(x,m=m,c=c):
    v=x['vote_count']
    r=x['vote_average']
    #now remember the imdb formula 
    return (v/(v+m) * r) + (m/(m+v) * c)


# In[49]:


#Aplicamos esto a nuestros datos con qualified_movies a través de la función apply().
qualified_movies['score']=qualified_movies.apply(imdbscore,axis=1)


# Ya realizada a nuestros datos, se le aplica la función sorted_values() para ordenar y que nos muestre las mejores películas

# In[48]:


# Ordenar películas según la puntuación calculada anteriormente
qualified_movies = qualified_movies.sort_values('score', ascending=False)

qualified_movies[['title', 'vote_count', 'vote_average', 'score']].head(10)


# # Sistema de recomendación colaborativo

# Usaremos la biblioteca Surprise para implementar SVD.

# In[29]:


reader = Reader()


# In[28]:


data_ratings = Dataset.load_from_df(df_ratings[['userId', 'movieId', 'rating']], reader)


# Creo un objeto svd para llamar al método de ajuste que hace que rmse sea menor y luego se transforma en dos matrices

# In[30]:


svd = SVD()


# In[31]:


trainset = data_ratings.build_full_trainset()
svd.fit(trainset)


# In[53]:


df_ratings[df_ratings['userId'] == 5]


# Realizamos una simulación de un usuario ('5'), y la movie ('104')

# In[54]:


svd.predict(5, 104)


# Predice 3,8 y esta cerca de su rating de 4, por lo que funciona como sistema de recomendación
