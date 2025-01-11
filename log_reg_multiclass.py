from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# https://towardsdatascience.com/multiclass-logistic-regression-from-scratch-9cc0007da372

class MultiClassLogisticRegression:

    def __init__(self, num_features, num_classes, max_iter=1000, lr = 0.01):
        self.W = np.zeros((num_classes, num_features))
        self.b = np.zeros((num_classes, 1))
        self.max_iter = max_iter
        self.learning_rate = lr
        self.losses = []


    def __sigmoid(self, z):
        return 1/(1 + np.exp(-z))
    
    def __softmax(self, z): # z is a vector
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    def __gradient(self, X, y, y_hat):
        m = X.shape[0]
        dW = (1 / m) * np.dot((y_hat.T - y).T, X)
        db = (1 / m) * np.sum(y_hat.T - y, axis=0, keepdims=True).T
        return dW, db

    def __gradient_descent(self, X, y):
        for i in range(self.max_iter):
            y_hat = self.__forward_pass(X)
            dW, db = self.__gradient(X, y, y_hat)
            self.W -= self.learning_rate * dW
            self.b -= self.learning_rate * db
            loss = self.__loss(y, y_hat)
            self.losses.append(loss)
            if i % 100 == 0:
                print(f"Iteration {i}, Loss: {loss:.4f}")

    def __loss(self, y, y_hat):
        # term1 = y * np.log(y_hat)
        # term2 = (1 - y) * np.log(1 - y_hat)
        # return -(term1 + term2)
        m = y.shape[0]
        return -np.sum(y * np.log(y_hat.T + 1e-15)) / m
    
    def __forward_pass(self, X):
        z = np.dot(self.W, X.T) + self.b
        return self.__softmax(z)


    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)

        encoder = OneHotEncoder(sparse_output=False)
        y_one_hot = encoder.fit_transform(y)

        self.__gradient_descent(X, y_one_hot)


    def predict(self, X):
        X = np.array(X)
        y_hat = self.__forward_pass(X)
        return np.argmax(y_hat, axis=0)


X = load_iris().data
y = load_iris().target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)


num_features = X_train.shape[1]
num_classes = len(np.unique(y))
model = MultiClassLogisticRegression(num_features, num_classes)
model.fit(X_train, y_train)
model.__forward_pass()

out = model.predict(X_test)

print(y_test, out)
