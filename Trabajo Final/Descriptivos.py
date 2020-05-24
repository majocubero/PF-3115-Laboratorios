#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 18:46:45 2020

@author: osvaldo
"""

import pandas as pd


#leyendo los datos 
datos=pd.read_csv("/home/osvaldo/CURSOS/Machine Learning/Trabajo Final/Datos/bank-full.csv", sep=";")

# cambiando el -1 por 0 de la columna pdays
datos['pdays']= datos['pdays'].replace(-1,0)

#Función para crear tablas cruzadas 
def tablas(columna,apredecir):
    tb=pd.crosstab(columna, apredecir)
    tb['%_acep']=(tb.yes)/(tb.yes + tb.no)*100
    return(tb.sort_values(by='%_acep', ascending=False))

# % aceptación por tipo de trabajo 
tablas(datos.job,datos.y)

#% aceptación por estado civil 
tablas(datos.marital, datos.y)

# % aceptación por grado de educacion 
tablas(datos.education, datos.y)

# % aceptacion si ha quedado default en el pasado 
tablas(datos.default,datos.y)

# % aceptacion personas con hipoteca 
tablas(datos.housing,datos.y)

# % aceptacion personas con prestamos personales
tablas(datos.loan,datos.y)

# Tablas con promedios 
print(pd.crosstab(datos.y,datos.y, values= datos.balance, aggfunc='mean' ).round(2))

print(pd.crosstab(datos.y,datos.y, values= datos.duration, aggfunc='mean' ).round(2))

print(pd.crosstab(datos.y,datos.y, values= datos.campaign, aggfunc='mean' ).round(2))

print(pd.crosstab(datos.y,datos.y, values= datos.pdays, aggfunc='mean' ).round(2))

