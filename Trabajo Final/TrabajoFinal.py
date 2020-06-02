#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 18:43:02 2020

@author: osvaldo
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report



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
t1=tablas(Training.job,Training.y)

#% aceptación por estado civil 
t2 = tablas(Training.marital, Training.y)

# % aceptación por grado de educacion 
t3=tablas(Training.education, Training.y)

# % aceptacion si ha quedado default en el pasado 
t4 = tablas(Training.default,Training.y)

# % aceptacion personas con hipoteca 
t5 = tablas(Training.housing,Training.y)

# % aceptacion personas con prestamos personales
t6 = tablas(Training.loan,Training.y)

# Tablas con promedios 
t7 = pd.crosstab(Training.y,Training.y, values= Training.balance, aggfunc='mean' ).round(2)

t8= pd.crosstab(Training.y,Training.y, values= Training.duration, aggfunc='mean' ).round(2)

t9= pd.crosstab(Training.y,Training.y, values= Training.campaign, aggfunc='mean' ).round(2)

t10= pd.crosstab(Training.y,Training.y, values= Training.pdays, aggfunc='mean' ).round(2)

### Regresión Logística ---------------------------------------------------------------------------------
# Se cargan los datos y se quitan las columnas que no nos funcionan 
X_train=pd.read_csv("/home/osvaldo/Documents/CURSOS/Machine Learning and Statistics/Trabajo Final /Datos/X_train.csv", sep=",")
X_train=X_train.drop(['Unnamed: 0', 'day', 'month'], axis=1)      
y_train=pd.read_csv("/home/osvaldo/Documents/CURSOS/Machine Learning and Statistics/Trabajo Final /Datos/y_train.csv", sep=",")
y_train=y_train.drop(['Unnamed: 0'], axis=1)  
X_test=pd.read_csv("/home/osvaldo/Documents/CURSOS/Machine Learning and Statistics/Trabajo Final /Datos/X_test.csv", sep=",")
X_test=X_test.drop(['Unnamed: 0', 'day', 'month'], axis=1)  
y_test=pd.read_csv("/home/osvaldo/Documents/CURSOS/Machine Learning and Statistics/Trabajo Final /Datos/y_test.csv", sep=",")
y_test=y_test.drop(['Unnamed: 0'], axis=1)  

#Convirtiendo en Dummies las variables categoricas de X_train
job1=pd.get_dummies(X_train['job'])
marital1=pd.get_dummies(X_train['marital'])
education1=pd.get_dummies(X_train['education'])
contact1=pd.get_dummies(X_train['contact'])
poutcome1=pd.get_dummies(X_train['poutcome'])

# cambiando loans, housing y default por 1 y 0 de X_train
X_train['loan']= X_train['loan'].replace('yes', 1)
X_train['loan']= X_train['loan'].replace('no', 0)

X_train['housing']= X_train['housing'].replace('yes', 1)
X_train['housing']= X_train['housing'].replace('no', 0)

X_train['default']= X_train['default'].replace('yes', 1)
X_train['default']= X_train['default'].replace('no', 0)

# Quitando variables que no se utilizaran de X_train
X_train=X_train.drop(['job', 'marital', 'education', 'contact', 'poutcome'], axis=1) 

# Uniendo las dummies a X_train 
X_train= pd.concat([X_train, job1, marital1, education1, contact1, poutcome1], axis=1) 

#Convirtiendo en Dummies las variables categoricas de X_test
job1=pd.get_dummies(X_test['job'])
marital1=pd.get_dummies(X_test['marital'])
education1=pd.get_dummies(X_test['education'])
contact1=pd.get_dummies(X_test['contact'])
poutcome1=pd.get_dummies(X_test['poutcome'])

# cambiando loans, housing y default por 1 y 0 de X_test
X_test['loan']= X_test['loan'].replace('yes', 1)
X_test['loan']= X_test['loan'].replace('no', 0)

X_test['housing']= X_test['housing'].replace('yes', 1)
X_test['housing']= X_test['housing'].replace('no', 0)

X_test['default']= X_test['default'].replace('yes', 1)
X_test['default']= X_test['default'].replace('no', 0)

# Quitando variables que no se utilizaran de X_test
X_test=X_test.drop(['job', 'marital', 'education', 'contact', 'poutcome'], axis=1) 

# Uniendo las dummies a X_test 
X_test= pd.concat([X_test, job1, marital1, education1, contact1, poutcome1], axis=1) 

lr = LogisticRegression(solver='liblinear', random_state=0)
lr.fit(X_train, y_train)
prediccion = lr.predict(X_test)

prediccion = pd.DataFrame(prediccion)
resultado =pd.concat([y_test, prediccion], axis=1)
resultado=resultado.rename(columns={'y':'verdadero', 0:'predicho'})
print(pd.crosstab(resultado.predicho, resultado.verdadero))
print(classification_report(y_test, prediccion))
