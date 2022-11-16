from nlpUtilidades import clean_text
import warnings
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer


with open('models/model_pickle_SVM_Ofensivo_stopw_3','rb') as f_ofensivo:
    clf_ofensivo = pickle.load(f_ofensivo)
with open('vectorizer/vectorizer_pickle_SVM_Ofensivo_stopw_3','rb') as x_ofensivo:
    tfidf_ofensivo  = pickle.load(x_ofensivo)

with open('models/model_pickle_SVM_Vulgar_stopw_3','rb') as f_vulgar:
    clf_vulgar  = pickle.load(f_vulgar)
with open('vectorizer/vectorizer_pickle_SVM_Vulgar_stopw_3','rb') as x_vulgar:
    tfidf_vulgar   = pickle.load(x_vulgar)

with open('models/model_pickle_SVM_Agresivo_stopw_3','rb') as f_agresivo:
    clf_agresivo = pickle.load(f_agresivo)
with open('vectorizer/vectorizer_pickle_SVM_Agresivo_stopw_3','rb') as x_agresivo:
    tfidf_agresivo = pickle.load(x_agresivo)

def clasificar_texto_ofensivo(texto):
    dato = clean_text(texto)
    vec = tfidf_ofensivo.transform([dato])
    aux = clf_ofensivo.predict(vec)
    return int(aux[0])

def clasificar_texto_agresivo(texto):
    dato = clean_text(texto)
    vec = tfidf_agresivo.transform([dato])
    aux = clf_agresivo.predict(vec)
    return int(aux[0])

def clasificar_texto_vulgar(texto):
    dato = clean_text(texto)
    vec = tfidf_vulgar.transform([dato])
    aux = clf_vulgar.predict(vec)
    return int(aux[0])
