#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 18:43:02 2020

@author: osvaldo
"""
import data_processor
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import sys


def cargar_datos_de_modelos():
    X_train= pd.read_csv("./Datos/X_train.csv", sep=",").drop(['Unnamed: 0', 'day', 'month'], axis=1)      
    y_train= pd.read_csv("./Datos/y_train.csv", sep=",").drop(['Unnamed: 0'], axis=1)  
    X_test= pd.read_csv("./Datos/X_test.csv", sep=",").drop(['Unnamed: 0', 'day', 'month'], axis=1)  
    y_test= pd.read_csv("./Datos/y_test.csv", sep=",").drop(['Unnamed: 0'], axis=1)
    return X_train, y_train, X_test, y_test

def regresion_logistica(X_train, y_train, X_test, y_test):
    print("--- REGRESION LOGISTICA \n")
    lr = LogisticRegression(solver='liblinear', random_state=0)
    lr.fit(X_train, y_train.values.ravel())
    prediccion = lr.predict(X_test)
    
    prediccion = pd.DataFrame(prediccion)
    resultado =pd.concat([y_test, prediccion], axis=1)
    resultado=resultado.rename(columns={'y':'verdadero', 0:'predicho'})
    
    imprimir_resultados(resultado.predicho, resultado.verdadero, y_test, prediccion)

def arboles_de_desicion(X_train, y_train, X_test, y_test):
    print("--- ARBOLES DE DECISION \n")
    tree_clf = tree.DecisionTreeClassifier(max_depth=40)
    tree_clf.fit(X_train, y_train.values.ravel())
    prediccion = tree_clf.predict(X_test)
    prediccion = pd.DataFrame(prediccion)
    resultado =pd.concat([y_test, prediccion], axis=1)
    resultado=resultado.rename(columns={'y':'verdadero', 0:'predicho'})
    
    imprimir_resultados(resultado.predicho, resultado.verdadero, y_test, prediccion)

def k_vecinos_mas_cercanos(X_train, y_train, X_test, y_test):
    print("--- K VECINOS MAS CERNANOS \n")
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    classifier = KNeighborsClassifier(n_neighbors=191)
    classifier.fit(X_train, y_train.values.ravel())
    
    prediccion = classifier.predict(X_test)
    prediccion = pd.DataFrame(prediccion)
    resultado =pd.concat([y_test, prediccion], axis=1)
    resultado=resultado.rename(columns={'y':'verdadero', 0:'predicho'})
    
    imprimir_resultados(resultado.predicho, resultado.verdadero, y_test, prediccion)

def imprimir_resultados(resut_predicho, result_verdadero, y_test, prediccion):
    print("1. Tabla de confusión: \n", pd.crosstab(resut_predicho, result_verdadero))
    print("\n2. Reporte de clasificación: \n", classification_report(y_test, prediccion))

def main():
    # 1. Sin agrupar datos
    print("A. Sin agrupación: ----------------------------------- \n")
    X_train, y_train, X_test, y_test = cargar_datos_de_modelos()
    X_train, X_test = data_processor.procesar_datos(X_train, X_test)
    regresion_logistica(X_train, y_train, X_test, y_test)
    arboles_de_desicion(X_train, y_train, X_test, y_test)
    k_vecinos_mas_cercanos(X_train, y_train, X_test, y_test)
     
    ####Cambiando un poco los datos 
    print("\nB. Agrupando por job --------------------------------- \n")
    X_train, y_train, X_test, y_test = cargar_datos_de_modelos() 
    X_train, X_test = data_processor.procesar_datos_agrupando_job(X_train, X_test)
    
    regresion_logistica(X_train, y_train, X_test, y_test)
    arboles_de_desicion(X_train, y_train, X_test, y_test)
    k_vecinos_mas_cercanos(X_train, y_train, X_test, y_test)
    
    ################### cambiando datos vol 2#####################
    print("\nC. Agrupando por job, p_outcome y contact-------------")
    X_train, y_train, X_test, y_test = cargar_datos_de_modelos() 
    X_train, X_test = data_processor.procesar_datos_agrupando_job_poutcome_contact(X_train, X_test)
    
    regresion_logistica(X_train, y_train, X_test, y_test)
    arboles_de_desicion(X_train, y_train, X_test, y_test)
    k_vecinos_mas_cercanos(X_train, y_train, X_test, y_test)
    
if __name__ == "__main__":
    main(*sys.argv[1:])