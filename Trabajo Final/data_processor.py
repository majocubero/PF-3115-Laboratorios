# -*- coding: utf-8 -*-

import pandas as pd


def procesar_datos(X_train, X_test): 
    return procesar_datos_train(X_train), procesar_datos_test(X_test)
    
def procesar_datos_train(X_train):
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
    return X_train
    
def procesar_datos_test(X_test):
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
    return X_test

'''Observando el conteo de la variable job se puede agrupar unemployed, housemaid, student y 
unknown en otros'''

def procesar_datos_agrupando_job(X_train, X_test):
    return procesar_datos_train_agrupando_job(X_train), procesar_datos_test_agrupando_job(X_test)
    
    
def procesar_datos_train_agrupando_job(X_train):
    ########## Cambiando la variable job #################
    X_train["job"]=X_train["job"].replace(["unemployed", "housemaid", "student", "unknown"], "others")
    X_train["job"]=X_train["job"].replace(["blue-collar", "management"], "bluecollar")
    X_train["job"]=X_train["job"].replace(["technician", "admin.", "services"], "tech-serv")
    X_train["job"]=X_train["job"].replace(["self-employed", "entrepreneur"], "entrepeneur")

    ####Quitando la variable poutcome#################
    X_train=X_train.drop('poutcome', axis=1)
    
    #Convirtiendo en Dummies las variables categoricas de X_train
    job1=pd.get_dummies(X_train['job'])
    marital1=pd.get_dummies(X_train['marital'])
    contact1=pd.get_dummies(X_train['contact'])
    education1=pd.get_dummies(X_train['education'])
    
    # cambiando loans, housing y default por 1 y 0 de X_train
    X_train['loan']= X_train['loan'].replace('yes', 1)
    X_train['loan']= X_train['loan'].replace('no', 0)
    
    X_train['housing']= X_train['housing'].replace('yes', 1)
    X_train['housing']= X_train['housing'].replace('no', 0)
    
    X_train['default']= X_train['default'].replace('yes', 1)
    X_train['default']= X_train['default'].replace('no', 0)
    
    # Quitando variables que no se utilizaran de X_train
    X_train=X_train.drop(['job', 'marital', 'contact', 'education'], axis=1) 
    
    # Uniendo las dummies a X_train 
    X_train= pd.concat([X_train, job1, marital1, contact1, education1], axis=1) 
    return X_train
    
def procesar_datos_test_agrupando_job(X_test):
    ########## Cambiando la variable job #################
    X_test["job"]=X_test["job"].replace(["unemployed", "housemaid", "student", "unknown"], "others")
    X_test["job"]=X_test["job"].replace(["blue-collar", "management"], "bluecollar")
    X_test["job"]=X_test["job"].replace(["technician", "admin.", "services"], "tech-serv")
    X_test["job"]=X_test["job"].replace(["self-employed", "entrepreneur"], "entrepeneur")
    
    ####Quitando la variable poutcome#################
    X_test=X_test.drop('poutcome', axis=1)  
    
    #Convirtiendo en Dummies las variables categoricas de X_test
    job1=pd.get_dummies(X_test['job'])
    marital1=pd.get_dummies(X_test['marital'])
    contact1=pd.get_dummies(X_test['contact'])
    education1=pd.get_dummies(X_test['education'])
    
    # cambiando loans, housing y default por 1 y 0 de X_test
    X_test['loan']= X_test['loan'].replace('yes', 1)
    X_test['loan']= X_test['loan'].replace('no', 0)
    
    X_test['housing']= X_test['housing'].replace('yes', 1)
    X_test['housing']= X_test['housing'].replace('no', 0)
    
    X_test['default']= X_test['default'].replace('yes', 1)
    X_test['default']= X_test['default'].replace('no', 0)
    
    # Quitando variables que no se utilizaran de X_test
    X_test= X_test.drop(['job', 'marital', 'contact', 'education'], axis=1) 
    
    # Uniendo las dummies a X_test 
    X_test= pd.concat([X_test, job1, marital1, contact1, education1], axis=1) 
    return X_test

'''Observando el conteo de la variable job se puede agrupar unemployed, housemaid, student y 
unknown en otros'''
def procesar_datos_agrupando_job_poutcome_contact(X_train, X_test):
   return procesar_datos_train_agrupando_job_poutcome_contact(X_train), procesar_datos_test_agrupando_job_poutcome_contact(X_test)
    
def procesar_datos_train_agrupando_job_poutcome_contact(X_train):
    X_train["job"].value_counts()
    X_train["education"].value_counts()
    X_train["poutcome"].value_counts()
    X_train["contact"].value_counts()
    ########## Cambiando la variable job #################
    X_train["job"]=X_train["job"].replace(["unemployed", "housemaid", "student", "unknown"], "others")
    X_train["job"]=X_train["job"].replace(["blue-collar", "management"], "bluecollar")
    X_train["job"]=X_train["job"].replace(["technician", "admin.", "services"], "tech-serv")
    X_train["job"]=X_train["job"].replace(["self-employed", "entrepreneur"], "entrepeneur")
    
    ####### Cambiando la variable poutcome ############
    X_train["poutcome"]=X_train["poutcome"].replace(["unknown", "failure", "other"], "other_poutcome")
    
     ###### Cambiando la variable contact #############
    X_train["contact"]=X_train["contact"].replace(["unknown", "telephone"], "other_contact")
    
    #Convirtiendo en Dummies las variables categoricas de X_train
    job1=pd.get_dummies(X_train['job'])
    marital1=pd.get_dummies(X_train['marital'])
    contact1=pd.get_dummies(X_train['contact'])
    education1=pd.get_dummies(X_train['education'])
    poutcome1=pd.get_dummies(X_train['poutcome'])
    
    # cambiando loans, housing y default por 1 y 0 de X_train
    X_train['loan']= X_train['loan'].replace('yes', 1)
    X_train['loan']= X_train['loan'].replace('no', 0)
    
    X_train['housing']= X_train['housing'].replace('yes', 1)
    X_train['housing']= X_train['housing'].replace('no', 0)
    
    X_train['default']= X_train['default'].replace('yes', 1)
    X_train['default']= X_train['default'].replace('no', 0)
    
    # Quitando variables que no se utilizaran de X_train
    X_train=X_train.drop(['job', 'marital', 'contact', 'education', 'poutcome'], axis=1) 
    
    # Uniendo las dummies a X_train 
    X_train= pd.concat([X_train, job1, marital1, contact1, education1, poutcome1], axis=1) 
    return X_train
    
def procesar_datos_test_agrupando_job_poutcome_contact(X_test):
    X_test["job"]=X_test["job"].replace(["unemployed", "housemaid", "student", "unknown"], "others")
    X_test["job"]=X_test["job"].replace(["blue-collar", "management"], "bluecollar")
    X_test["job"]=X_test["job"].replace(["technician", "admin.", "services"], "tech-serv")
    X_test["job"]=X_test["job"].replace(["self-employed", "entrepreneur"], "entrepeneur")
    
    X_test["poutcome"]=X_test["poutcome"].replace(["unknown", "failure", "other"], "other_poutcome")
    
    X_test["contact"]=X_test["contact"].replace(["unknown", "telephone"], "other_contact")
    
    #Convirtiendo en Dummies las variables categoricas de X_test
    job1=pd.get_dummies(X_test['job'])
    marital1=pd.get_dummies(X_test['marital'])
    contact1=pd.get_dummies(X_test['contact'])
    education1=pd.get_dummies(X_test['education'])
    poutcome1=pd.get_dummies(X_test['poutcome'])
    
    # cambiando loans, housing y default por 1 y 0 de X_test
    X_test['loan']= X_test['loan'].replace('yes', 1)
    X_test['loan']= X_test['loan'].replace('no', 0)
    
    X_test['housing']= X_test['housing'].replace('yes', 1)
    X_test['housing']= X_test['housing'].replace('no', 0)
    
    X_test['default']= X_test['default'].replace('yes', 1)
    X_test['default']= X_test['default'].replace('no', 0)
    
    # Quitando variables que no se utilizaran de X_test
    X_test=X_test.drop(['job', 'marital', 'contact', 'education', 'poutcome'], axis=1) 
    
    # Uniendo las dummies a X_test 
    X_test= pd.concat([X_test, job1, marital1, contact1, education1, poutcome1], axis=1)
    return X_test
    
    