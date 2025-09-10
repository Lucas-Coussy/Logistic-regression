import numpy as np

class Logisticregression():

    def __init__(self, learning_rate=0.01, epochs=100, threshold=0.5):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.threshold = threshold
        self.hist = []
        self.costs = []
        
    def X_B(self, X):
        return self.beta[0] + np.dot(X, self.beta[1:])
    
    def P(self, z):
        return 1./(1.+np.exp(-z))
    
    def predict(self, X):
        proba = self.P(self.X_B(X))
        return np.where(proba >= self.threshold, 1, 0)
    
    def cost(self, y, proba):
        return np.dot(y,np.log(proba)) + np.dot((1-y),np.log(1-proba))
        #return np.dot(y,self.X_B(X)) - np.log(1 + np.exp(self.X_B(X))) 
    
    def gradient_decent(self, X, y):
        for i in range(self.epochs):
            proba = self.P(self.X_B(X))
            proba = np.where(proba < 1e-6, 1e-6, proba) #avoid problem with log giving -infinity
            proba = np.where(proba > 1 - 1e-6, 1 - 1e-6, proba) #avoid problem with log giving -infinity
            errors = y - proba
            self.beta[1:] += self.learning_rate * X.T.dot(errors)
            self.beta[0] += self.learning_rate * errors.sum()
            yield self.cost(y, proba)

    def fit(self, X, y):
        self.beta = np.random.normal(loc=0.0, scale=0.1, size=1 + X.shape[1])
        self.hist.append(self.beta.copy())

        for cost in self.gradient_decent(X, y):
            self.hist.append(self.beta.copy())
            self.costs.append(cost)
        return self
