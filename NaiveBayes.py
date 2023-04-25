import math

class Naive_bayes:
    def __init__(self):
        self.py= 0
        self.medias=0
        self.dp=0
        self.px =0

    def calcula_Py(self, y):
        quantidade = {}
        prob = {}

        for i in y:
            if i in quantidade:
                quantidade[i] +=1
            else:
                quantidade[i] = 1

        for k,v in quantidade.items():
            prob[k] = v/len(y)

        return prob

    def separa_classe(self, X, y):
        retorno = {}
        for i in range(len(y)):
            retorno[y[i]] = []
        
        for i in range(len(y)):
            retorno[y[i]].append(X[i])

        return retorno

    def calcula_media(self, X):
        medias = {}
        soma = 0
        cont = 0
        i = 0
        
        
        while i < len(X[0]):
            for x in X:
                soma += x[i]
                cont += 1
            
            media = soma/cont
            medias[i] =  media
            i +=1
            soma = 0
            media = 0
            cont = 0
        return medias
        
    def calcula_DP(self, X):
        dp = {}
        soma = 0
        cont = 0
        i = 0
        medias = self.calcula_media(X)

        while i < len(X[0]):
            for x in X:
                #calcula a diferenca do intem com a media
                soma += (x[i] - medias[i])**2
                cont += 1
            
            variancia = soma/cont
            dp[i] =  variancia**(1/2)
            i +=1
            soma = 0
            media = 0
            cont = 0
        return dp

    def calcula_Px_gerla(self, X):
        px = []
        probs = []
        medias = self.calcula_media(X)
        dps = self.calcula_DP(X)

        for i in range(len(X)):
            for j in range(len(X[i])):
                xi = X[i][j]
                u = medias[j]
                s = dps[j]
                pi = math.pi 
                e = math.e

                p1 = 1/(s*math.sqrt(2*pi))
                z = (xi-u)/s
                expoente = (-1/2)*(z**2)
                prob = p1 * e**expoente
                probs.append(prob)
            px.append(probs)
            probs = []
        
        return px

    def calcula_media_classe(self, X, y):
        medias = {}
        X_separado = self.separa_classe(X, y)

        for k, v in X_separado.items():
            medias[k] = self.calcula_media(v)

        return medias    

    def calcula_dp_classe(self, X, y):
        dp = {}
        X_separado = self.separa_classe(X, y)

        for k, v in X_separado.items():
            dp[k] = self.calcula_DP(v)

        return dp    

    def calcula_Px(self, xi, media, dp):
        u = media
        s = dp

        pi = math.pi 
        e = math.e
        p1 = 1/(s*math.sqrt(2*pi))
        z = (xi-u)/s
        expoente = (-1/2)*(z**2)
        prob = p1 * e**expoente

        return prob

    def fit(self, X, y):
        self.py = self.calcula_Py(y)
        self.medias = self.calcula_media_classe(X, y)
        self.dp = self.calcula_dp_classe(X, y)
        self.px = self.calcula_Px_gerla(X)

    def predict(self, xi):
        probs = {}
        conta = 0
        
        for k, v in self.py.items():
            for i in range(len(self.medias[k])):
                conta +=  math.log(self.calcula_Px(xi[i],self.medias[k][i],self.dp[k][i]))
            probs[k] = conta + math.log(self.py[k])
        
        for v in probs.values():
            maior = v
        retorno = 'erro'
        for k, v in probs.items():
            if v > maior:
                maior = v
                retorno = k

        return retorno
        