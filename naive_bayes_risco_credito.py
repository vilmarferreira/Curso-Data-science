#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 23:52:37 2018

@author: vilmarferreira
"""

import pandas as pd
base = pd.read_csv('risco-credito.csv')
previsores = base.iloc[:,0:4].values
classe = base.iloc[:,4]


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
previsores[:,0] = labelencoder.fit_transform(previsores[:,0])
previsores[:,1] = labelencoder.fit_transform(previsores[:,1])
previsores[:,2] = labelencoder.fit_transform(previsores[:,2])
previsores[:,3] = labelencoder.fit_transform(previsores[:,3])

from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()

#treinamento do algoritmo
classificador.fit(previsores,classe)

#
## verificacao- analise de um novo registro
resultado = classificador.predict([[0,0,1,2],[3,0,0,0]])

print(classificador.classes_)
print(classificador.class_count_)
print(classificador.class_prior_)

