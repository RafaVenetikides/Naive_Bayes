class Naive_bayes:
    def __init__(self):
        pass

    def Py(self, y):
        quantidade = {}
        prob = {}

        for i in y:
            if i in quantidade:
                quantidade[i] +=1
            else:
                quantidade[i] = 1

        for v in quantidade.values():
            prob[v] = v/len(y)

        print(prob)



