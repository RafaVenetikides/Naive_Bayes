import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split

def trainNB(data, labels):
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

    # dando outro nome para a variavel para utilizar mais tarde
    totalLabels = num_linhas
    somaLabels = []

    # calculando a probabilidade por resultado e guardando no vetor
    for i in range(len(numLabels)):
        sum = 0
        for j in labels:
            if j == numLabels[i]:
                sum += 1
        somaLabels.append(sum);
        pLabels[i] = sum / totalLabels

    # salvando todos os possiveis valores para os atributos em um vetor

    numData = []

    for i in range(len(data)):
        for j in range(len(data[0])):
            if data[i][j] not in numData:
                numData.append(data[i][j])

    numData.sort()

    # criando uma matriz "matrizSoma" em que as dimensoes representam: (resultados X numero de atributos X numero de valores para os atributos)
    # pData (numero de atributos X numero de valores para os atributos X resultados)
    matrizSoma = np.zeros((len(numLabels), num_colunas, len(numData)), dtype=float)
    pData = np.zeros((num_colunas, len(numData), len(numLabels)), dtype=float)

    database = np.hstack((data, labels.reshape(-1, 1)))

    # Realizando a soma dos valores por posicao, tipo de tom e resultado

    for label in range(len(numLabels)):
        for instance in range(num_linhas):
            for attribute in range(num_colunas):
                attribute_value = data[instance][attribute]
                value_index = numData.index(attribute_value)
                if labels[instance] == numLabels[label]:
                    matrizSoma[label][attribute][value_index] += 1

    # Com a soma, fazendo a divisão para obter as probabilidades

    for label in range(len(numLabels)):
        for attribute in range(num_colunas):
            for value_index in range(len(numData)):
                attribute_label_count = matrizSoma[label][attribute][value_index]
                label_count = somaLabels[label]
                pData[attribute][value_index][label] = attribute_label_count / label_count

    return pLabels, pData, numLabels, numData


def predictNB(pLabels, pData, instance, numLabels, numData):
    # num_linhas = quantidade de atributos por instancia
    num_linhas = len(instance)

    # vetor que guardara as probabilidades para cada resultado
    pResults = [0] * len(numLabels)

    # calculando a probabilidade para cada resultado
    for result in range(len(pResults)):
        # probabilidade inicial eh a probabilidade do resultado em si
        p = pLabels[result]

        # para cada atributo, calcula a probabilidade do valor do atributo dado o resultado
        for attribute in range(num_linhas):
            attribute_value = instance[attribute]
            for i in range(len(numData)):
                if np.any(numData[i] == attribute_value):
                    p *= pData[attribute][i][result]

        # adiciona a probabilidade do resultado na lista de probabilidades
        pResults[result] = p

    # retorna o resultado com maior probabilidade
    return numLabels[pResults.index(max(pResults))]


X, y = load_breast_cancer(return_X_y=True)

X, X_val, y, y_val = train_test_split(X, y, train_size=0.8, random_state=42, stratify=y)

pLabels, pData, numLabels, numData = trainNB(X, y)

certos = 0
cont = 0
erros = 0

for predict in range(len(y_val)):
    try:
        if predictNB(pLabels, pData, X_val[predict], numLabels, numData) == y_val[predict]:
            certos += 1
        cont += 1
    except:
        erros += 1


print(f'acurácia: {certos/cont}')
