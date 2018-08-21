# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd 
base = pd.read_csv('credit-data.csv')
base.describe()

##Tratamento de valores inconsistentes 
##localizar idade menor que 0
base.loc[base['age']<0]
##apagar coluna :
base.drop('age',1,inplace=True)
##Apagar somente registros com problema 
base.drop(base[base.age <0].index, inplace = True)

##Preencher manualmente
#preencher os valores com a media 
base.mean()
base['age'].mean()
base['age'][base.age>0].mean()
base.loc[base.age<0, 'age']=40.92 


#valor nulo
pd.isnull(base['age'])

base.loc[pd.isnull(base['age'])]


#divisao 

#: todas as linhas do item 1:4 1 ao 4
previsores= base.iloc[:,1:4].values
classe = base.iloc[:,4].values



from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer = imputer.fit(previsores[:,0:3])
previsores[:, 0:3] = imputer.transform(previsores[:,0:3])


#aplicando padronizacao 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)
