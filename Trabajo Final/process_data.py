# -*- coding: utf-8 -*-

import pandas as pd
import sys

def procesar_datos(X_train, y_train, X_test, y_test): 
    procesar_datos_train(X_train, y_train)
    procesar_datos_test(X_test, y_test)
    
def procesar_datos_train(X_train, y_train):
    #Convirtiendo en Dummies las variables categoricas de X_train
    job1= pd.get_dummies(X_train['job'])
    marital1= pd.get_dummies(X_train['marital'])
    education1= pd.get_dummies(X_train['education'])
    contact1= pd.get_dummies(X_train['contact'])
    poutcome1= pd.get_dummies(X_train['poutcome'])
    
    # cambiando loans, housing y default por 1 y 0 de X_train
    X_train['loan']= X_train['loan'].replace('yes', 1)
    X_train['loan']= X_train['loan'].replace('no', 0)
    
    X_train['housing']= X_train['housing'].replace('yes', 1)
    X_train['housing']= X_train['housing'].replace('no', 0)
    
    X_train['default']= X_train['default'].replace('yes', 1)
    X_train['default']= X_train['default'].replace('no', 0)
    
    # Quitando variables que no se utilizaran de X_train
    X_train= X_train.drop(['job', 'marital', 'education', 'contact', 'poutcome'], axis=1) 
    
    # Uniendo las dummies a X_train 
    X_train= pd.concat([X_train, job1, marital1, education1, contact1, poutcome1], axis=1) 
    
def procesar_datos_test(X_test, y_test):
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

main = procesar_datos

if __name__ == "__main__":
    main(sys.argv[1])