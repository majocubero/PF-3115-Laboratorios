#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 18:43:02 2020

@author: osvaldo
"""

import pandas as pd
from sklearn.model_selection import train_test_split


#Todo lo comentado fue lo que se hizo para dividir los datos de manera balanceada 
'''
#leyendo los datos 
datos=pd.read_csv("/home/osvaldo/Documents/CURSOS/Machine Learning and Statistics/Trabajo Final /Datos/bank-full.csv", sep=";")

x=datos.drop(['y'], axis=1)
y=datos.y

# Dividiendo el data set en trainig set y test set 
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, stratify=y)

# Pruebas del balance de y 
Training=pd.concat([X_train, y_train], axis=1)
Testing=pd.concat([X_test, y_test], axis=1)

X_train.to_csv(r'/home/osvaldo/Documents/CURSOS/Machine Learning and Statistics/Trabajo Final /Datos/X_train.csv')
y_train.to_csv(r'/home/osvaldo/Documents/CURSOS/Machine Learning and Statistics/Trabajo Final /Datos/y_train.csv')
X_test.to_csv(r'/home/osvaldo/Documents/CURSOS/Machine Learning and Statistics/Trabajo Final /Datos/X_test.csv')
y_test.to_csv(r'/home/osvaldo/Documents/CURSOS/Machine Learning and Statistics/Trabajo Final /Datos/y_test.csv')

Training.to_csv(r'/home/osvaldo/Documents/CURSOS/Machine Learning and Statistics/Trabajo Final /Datos/Training.csv', index = False)
Testing.to_csv(r'/home/osvaldo/Documents/CURSOS/Machine Learning and Statistics/Trabajo Final /Datos/Testing.csv', index = False)

# cambiando el -1 por 0 de la columna pdays
datos['pdays']= datos['pdays'].replace(-1,0)
'''

# Cargando training que es de quien se van a obtener los descriptivos 
Training=pd.read_csv("/home/osvaldo/Documents/CURSOS/Machine Learning and Statistics/Trabajo Final /Datos/Training.csv", sep=",")

#Función para crear tablas cruzadas 
def tablas(columna,apredecir):
    tb=pd.crosstab(columna, apredecir)
    tb['%_acep']=(tb.yes)/(tb.yes + tb.no)*100
    return(tb.sort_values(by='%_acep', ascending=False))

# % aceptación por tipo de trabajo 
tablas(Training.job,Training.y)

#% aceptación por estado civil 
tablas(Training.marital, Training.y)

# % aceptación por grado de educacion 
tablas(Training.education, Training.y)

# % aceptacion si ha quedado default en el pasado 
tablas(Training.default,Training.y)

# % aceptacion personas con hipoteca 
tablas(Training.housing,Training.y)

# % aceptacion personas con prestamos personales
tablas(Training.loan,Training.y)

# Tablas con promedios 
print(pd.crosstab(Training.y,Training.y, values= Training.balance, aggfunc='mean' ).round(2))

print(pd.crosstab(Training.y,Training.y, values= Training.duration, aggfunc='mean' ).round(2))

print(pd.crosstab(Training.y,Training.y, values= Training.campaign, aggfunc='mean' ).round(2))

print(pd.crosstab(Training.y,Training.y, values= Training.pdays, aggfunc='mean' ).round(2))
