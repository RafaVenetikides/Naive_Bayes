# Importando as bibliotecas necessárias
import math
import collections

# Criando uma função para contar a frequência de palavras em um texto
def count_words(text):
    words = []
    for sentence in text:
        for word in sentence.split():
            words.append(word)
    return collections.Counter(words)

# Criando uma função para treinar o classificador Naive Bayes
def train_naive_bayes(training_data, training_labels):
    # Contando a frequência de palavras em cada classe
    class_word_counts = collections.defaultdict(lambda: collections.Counter())
    for x, y in zip(training_data, training_labels):
        class_word_counts[y] += count_words(x)
    
    # Contando o número de exemplos de cada classe
    class_counts = collections.Counter(training_labels)
    
    # Calculando a probabilidade de cada palavra para cada classe
    word_probs = {}
    for label in class_counts.keys():
        total_count = sum(class_word_counts[label].values())
        word_probs[label] = {word: math.log(count/total_count + 1) for word, count in class_word_counts[label].items()}
        
    # Calculando a probabilidade de cada classe
    total_examples = len(training_labels)
    class_probs = {label: math.log(count/total_examples) for label, count in class_counts.items()}
    
    return word_probs, class_probs

# Criando uma função para classificar novos exemplos
def classify_naive_bayes(example, word_probs, class_probs):
    # Calculando a probabilidade de cada classe para o exemplo
    probs = {}
    for label in class_probs.keys():
        prob = class_probs[label]
        for word in example.split():
            if word in word_probs[label]:
                prob += word_probs[label][word]
        probs[label] = prob
    
    # Retornando a classe com a maior probabilidade
    return max(probs, key=probs.get)

# Exemplo de uso
training_data = ['este filme é ótimo', 'gostei muito deste filme', 'o enredo foi emocionante', 'não gostei deste filme', 'achei o filme fraco']
training_labels = ['positivo', 'positivo', 'positivo', 'negativo', 'negativo']

word_probs, class_probs = train_naive_bayes(training_data, training_labels)
new_example = 'este filme foi incrível'
classification = classify_naive_bayes(new_example, word_probs, class_probs)
print(classification)
