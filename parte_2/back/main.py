# librería Natural Language Toolkit, usada para trabajar con textos 
import nltk
# Punkt permite separar un texto en frases.
nltk.download('punkt')
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('wordnet')

from typing import Optional
from fastapi import FastAPI
from joblib import load
import pandas as pd
from DataModel import DataModel
from DataModel_mejorado import DataModelMejorado

from fastapi.middleware.cors import CORSMiddleware

# Librerías para manejo de datos

import pandas as pd
pd.set_option('display.max_columns', 25) # Número máximo de columnas a mostrar
pd.set_option('display.max_rows', 50) # Numero máximo de filas a mostar
import numpy as np
np.random.seed(3301)
import pandas as pd
# Para preparar los datos
from sklearn.preprocessing import LabelEncoder, StandardScaler,MinMaxScaler
# Para crear el arbol de decisión 

# Para realizar la separación del conjunto de aprendizaje en entrenamiento y test.
from sklearn.model_selection import train_test_split
# Para evaluar el modelo
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import plot_confusion_matrix
# Para búsqueda de hiperparámetros
from sklearn.model_selection import GridSearchCV
# Para la validación cruzada
from sklearn.model_selection import KFold 
#Librerías para la visualización
import matplotlib.pyplot as plt
# Seaborn
import seaborn as sns
import sklearn
import seaborn as sns; sns.set()  # for plot styling

from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_samples, silhouette_score
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D # for 3D plots

from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering


import re, string, unicodedata
import contractions
import inflect
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer

from sklearn.neural_network import MLPClassifier


from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report, confusion_matrix, plot_precision_recall_curve
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

# Importar/ Exportar modelos
from joblib import dump, load
import joblib

import matplotlib.pyplot as plt



app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/test")
def read_root():
   datos = np.array([[0, "study interventions are Carboplatin", "stage iib uterine sarcoma diagnosis and patients must be entered no more than twelve weeks post operatively"]])
   df = pd.DataFrame(datos, columns = ["label", "study", "condition"])
   rta = transform(data = df)
   rta = generar_lista(rta)
   
   columns = generar_columnas()
   df = pd.DataFrame([rta[0]], columns = columns)
   print (df)
   
   model = load("assets/redes.joblib")
   result = model.predict(df)
   print("resultado" + str(result))

   return {"Hello": "World"}

def generar_lista(entrada):
   warnings=""
   dfStudy = pd.read_csv("./data/llaves_study.csv")
   dfStudy = dfStudy.to_numpy()
   dfConditions = pd.read_csv("./data/llaves_conditions.csv")
   dfConditions = dfConditions.to_numpy()
   rta = np.zeros(len(dfConditions)+len(dfStudy))
   listaact = list(dfStudy)
   for palabra in entrada[0]:
      if palabra in dfStudy:
         rta[listaact.index(palabra)]=1
      else:
         print("error en " + palabra)
         warnings+=palabra+","
   listaact = list(dfConditions)
   for palabra in entrada[1]:
      if palabra in dfConditions:
         rta[listaact.index(palabra)+len(dfStudy)]=1
      else:
         print("error en " + palabra)
         warnings+=palabra+","
   return([rta, warnings])

def generar_columnas():
   dfStudy = pd.read_csv("./data/llaves_study.csv")
   dfStudy = dfStudy.to_numpy()
   dfConditions = pd.read_csv("./data/llaves_conditions.csv")
   dfConditions = dfConditions.to_numpy()
   rta = ["" for x in range(len(dfConditions)+len(dfStudy))]
   for i in range(len(dfStudy)):
      rta[i]="study_"+str(i)
   for i in range(len(dfConditions)):
      rta[i+len(dfStudy)]="condition_"+str(i)
   return rta


@app.post("/predict")
def make_predictions(dataModel: DataModel):
   print( "\nDatamodel:"+str(dataModel) + "\n\n")
   print("dict():" +str(dataModel.dict())+"\n\n")
   print("dict().keys():" +str(dataModel.dict().keys())+"\n\n")
   
   print(str(dataModel.dict().keys()))
   df = pd.DataFrame(dataModel.dict(), columns=dataModel.dict().keys(), index=[0])
   rta = transform(data = df)
   rta = generar_lista(rta)
   
   columns = generar_columnas()
   df = pd.DataFrame([rta[0]], columns = columns)
   model = load("assets/redes.joblib")
   result = model.predict(df)
   return({"prediction":str(result[0]), "warnings":str(rta[1])})


@app.post("/evaluar")
def evaluar(dataModel: DataModelMejorado):

   #print( "\nDatamodel:"+str(dataModel) + "\n\n")
   #print("dict():" +str(dataModel.dict())+"\n\n")
   #print("dict().keys():" +str(dataModel.dict().keys())+"\n\n")
   columna_predict = "Life expectancy"
   df1 = pd.DataFrame(dataModel.dict(), columns=dataModel.dict().keys())
   df1.columns = dataModel.columns()
   x = df1.drop(columna_predict, axis=1)
   y = df1[columna_predict]
   
   #print( "\n x:" + str(x) + "\n")
   #print( "\n y:" + str(y) + "\n")
   
   model = load("assets/model.joblib")
   model.fit(x,y)
   rta = model.score(x,y)
   #print(str(rta))
   return({"R^2":rta})
    
    
def transform(data: pd.DataFrame(), y = None):
   data['condition'] = data['condition'].apply(contractions.fix) #Aplica la corrección de las contracciones
   data['study'] = data['study'].apply(contractions.fix) #Aplica la corrección de las contracciones
        
   data['words'] = data['condition'].apply(word_tokenize).apply(preprocessing) #Aplica la eliminación del ruido
   data['words_study'] = data['study'].apply(word_tokenize).apply(preprocessing) #Aplica la eliminación del ruido
        
   data['words'] = data['words'].apply(stem_and_lemmatize)
   data['words_study'] = data['words_study'].apply(stem_and_lemmatize)
        
   data['words'] = data['words'].apply(lambda x: ' '.join(map(str, x)))
   data['words_study'] = data['words_study'].apply(lambda x: ' '.join(map(str, x)))
        
   X_data_words, X_data_words_study, y_data = data['words'], data['words_study'] ,data['label']
   y_data = pd.to_numeric(y_data)
   print(y_data)
        
   #xdtf condition
   dummy_words = CountVectorizer(binary=True)
   X_dummy_words = dummy_words.fit_transform(X_data_words)
   print(X_dummy_words.shape)
   X_dummy_words.toarray()[0]
   xdtf = pd.DataFrame(X_dummy_words.toarray())
   xdtf=xdtf.add_prefix('condition_')
        
   #xdtf study
   dummy_words_study = CountVectorizer(binary=True)
   X_dummy_words_study = dummy_words_study.fit_transform(X_data_words_study)
   print(X_dummy_words_study.shape)
   X_dummy_words_study.toarray()[0]
   xdtf_study = pd.DataFrame(X_dummy_words_study.toarray())
   xdtf_study=xdtf_study.add_prefix('study_')

   #Esto nos va a servir mas adelante para interpretar los resultados de los modelos
   llaves_study = dummy_words_study.vocabulary_.keys()
   llaves_study = list(llaves_study)
   print("llaves_study"+str(llaves_study))

   #Esto nos va a servir mas adelante para interpretar los resultados de los modelos
   llaves_conditions = dummy_words.vocabulary_.keys()
   llaves_conditions = list(llaves_conditions)
   print("llaves_conditions"+str(llaves_conditions))
   
   return [llaves_study, llaves_conditions]
   data = xdtf_study.copy()
   print(data.shape)
   condition_columns = xdtf.columns

   for column in condition_columns:
      data[column] = xdtf[column]

   print(data.shape)
        
   print(data)
   return data
    
    
#Estos metodos se usan para limpiar los registrados pasados por parametro
def remove_non_ascii(words):
   new_words = []
   for word in words:
      new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
      new_words.append(new_word)
   return new_words

def to_lowercase(words):
   new_words = []
   for word in words:
      new_word = word.lower()
      if new_word != '':
            new_words.append(new_word)
   return new_words

def remove_punctuation(words):
   new_words = []
   for word in words:
      new_word = re.sub(r'[^\w\s]', '', word)
      if new_word != '':
            new_words.append(new_word)
   return new_words

def replace_numbers(words):
   p = inflect.engine()
   new_words = []
   if(words==None): return new_words
   for word in words:
      if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
      else:
            new_words.append(word)
   return new_words

stop_words = set(stopwords.words('english'))

def remove_stopwords(words):
   new_words =[]
   for word in words:
      if word not in stop_words:
            new_words.append(word)
   return new_words

def preprocessing(words):
   #print("1 - " + str(words) + "\n")
   words = to_lowercase(words)
   #print("2 - " + str(words) + "\n")
   words = replace_numbers(words)
   #print("3 - " + str(words) + "\n")
   words = remove_punctuation(words)
   #print("4 - " + str(words) + "\n")
   words = remove_non_ascii(words)
   #print("5 - " + str(words) + "\n")
   words = remove_stopwords(words)
   #print("6 - " + str(words) + "\n\n")
   return words



#Estos metodos se usan para normalizar



def stem_words(words):
   ls = LancasterStemmer()
   rta = [ls.stem(word) for word in words]
   return rta

def lemmatize_verbs(words):
   wnl = WordNetLemmatizer()
   rta = [wnl.lemmatize(word) for word in words]
   return rta


def stem_and_lemmatize(words):
   stems = stem_words(words)
   lemmas = lemmatize_verbs(words)
   return stems + lemmas
