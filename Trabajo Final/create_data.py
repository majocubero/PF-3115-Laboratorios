# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import sys

# 1. DIVIDIR LOS DATOS DE MANERA BALANCEADA



def dividir_datos():
    #leyendo los datos 
    datos=pd.read_csv("./Datos/bank-full.csv", sep=";")
    
    x=datos.drop(['y'], axis=1)
    y=datos.y
    
    # Dividiendo el data set en trainig set y test set 
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, stratify=y)
    
    # Pruebas del balance de y 
    Training=pd.concat([X_train, y_train], axis=1)
    Testing=pd.concat([X_test, y_test], axis=1)
    
    X_train.to_csv(r'./Datos/X_train.csv')
    y_train.to_csv(r'./Datos/y_train.csv')
    X_test.to_csv(r'./Datos/X_test.csv')
    y_test.to_csv(r'./Datos/y_test.csv')
    
    Training.to_csv(r'./Datos/Training.csv', index = False)
    Testing.to_csv(r'./Datos/Testing.csv', index = False)
    
    # cambiando el -1 por 0 de la columna pdays
    datos['pdays']= datos['pdays'].replace(-1,0)
    

# 2. CREAR TABLAS CRUZADAS

def crear_tablas_cruzadas(): 
    # Cargando training que es de quien se van a obtener los descriptivos 
    Training= pd.read_csv("./Datos/Training.csv", sep=",")
    # % aceptación por tipo de trabajo 
    t1= crear_tabla(Training.job,Training.y)
    
    #% aceptación por estado civil 
    t2 = crear_tabla(Training.marital, Training.y)
    
    # % aceptación por grado de educacion 
    t3= crear_tabla(Training.education, Training.y)
    
    # % aceptacion si ha quedado default en el pasado 
    t4 = crear_tabla(Training.default,Training.y)
    
    # % aceptacion personas con hipoteca 
    t5 = crear_tabla(Training.housing,Training.y)
    
    # % aceptacion personas con prestamos personales
    t6 = crear_tabla(Training.loan,Training.y)
    return t1, t2, t3, t4, t5, t6

#Función para crear tablas cruzadas 
def crear_tabla(columna,apredecir):
    tb=pd.crosstab(columna, apredecir)
    tb['%_acep']=(tb.yes)/(tb.yes + tb.no)*100
    return(tb.sort_values(by='%_acep', ascending=False))

def main():
    dividir_datos()
    t1, t2, t3, t4, t5, t6 = crear_tablas_cruzadas()
    
if __name__ == "__main__":
    main(*sys.argv[1:])

''' Se crean estas variables cruzadad para poder generar los archivos de excel que se utilizan como entrada de los modelos'''
t1, t2, t3, t4, t5, t6 = crear_tablas_cruzadas()