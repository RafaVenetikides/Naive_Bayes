import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_digits
import math
import collections
from tabulate import tabulate

X, y = load_digits(return_X_y=True)

# transferindo valores do load para variaveis para depois transformar em uma funcao

data = X
labels = y

# num_linhas = quantidade de instancias
# num_colunas = quantidade deatributos por instancia

num_linhas = len(data)
num_colunas = len(data[0])

# Colocando todos os possiveis resultados em uma matriz
numLabels = []
for i in labels:
    if i not in numLabels:
        numLabels.append(i)

numLabels.sort()

# criando um vetor onde o indice no numLabels eh o mesmo da probabilidade

pLabels = [0] * len(numLabels)

i = 0
j = 0

#dando outro nome para a variavel para utilizar mais tarde
totalLabels = num_linhas

# calculando a probabilidade por resultado e guardando no vetor
for i in range(len(numLabels)):
    sum = 0
    for j in labels:
        if j == numLabels[i]:
            sum += 1
    pLabels[i] = sum/totalLabels

#salvando todos os possiveis valores para os atributos em um vetor

numData = []

for i in range(len(data)):
    for j in range(len(data[0])):
        if data[i][j] not in numData:
            numData.append(data[i][j])


numData.sort()

# criando uma matriz "matrizSoma" em que as dimensoes representam: (resultados X numero de atributos X numero de valores para os atributos)

matrizSoma = np.zeros((len(numLabels), num_colunas, len(numData)), dtype=float)
pData = np.zeros((num_colunas, len(numData), len(numLabels)), dtype=float)

print(f"linhas: {len(pData)}, colunas: {len(pData[0])},profundidade: {len(pData[0][0])}")

print(numData)
print(numLabels)

database = np.hstack((data, labels.reshape(-1, 1)))

print(f'database linhas: {len(database)}, database colunas: {len(database[0])}')

# Realizando a soma dos valores por posicao, tipo de tom e resultado

"""for label in range(len(numLabels)):
    vetsoma = [[0] * len(numData) for n in range(num_colunas)]
    for databaseColuna in range(len(database[label]) -1):
        for databaseLinha in range(len(database)):
            for tom in range(len(numData)):
                if database[databaseLinha][-1] == numLabels[label]:
                    vetsoma[databaseColuna][tom] += 1
        matrizSoma[label] = vetsoma"""

for label in range(len(numLabels)):
    for instance in range(num_linhas):
        for attribute in range(num_colunas):
            attribute_value = data[instance][attribute]
            value_index = numData.index(attribute_value)
            if labels[instance] == numLabels[label]:
                matrizSoma[label][attribute][value_index] += 1

print(matrizSoma)

# MATRIZ GERADA CORRETAMENTE!!!
