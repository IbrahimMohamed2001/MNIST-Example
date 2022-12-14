import numpy as np
import matplotlib.pyplot as plt
from os import listdir, mkdir
from os.path import isfile, join, isdir
import plotly.graph_objects as go
from time import time


class Layer:
    def __init__(self, neurons, activation, activation_drev):
        self.neurons = neurons
        self.activation = activation
        self.activation_drev = activation_drev


class NeuralNetwork:
    def __init__(self, X_train, X_cv, X_test, y_train, y_cv, y_test, layers):
        self.m = X_train.shape[0]
        self.epsilon = 1e-8

        self.X_train_norm, self.miu, self.var = self.data_normalize(X_train)
        self.X_test_norm = (X_test - self.miu) / np.sqrt(self.var + 1e-8)
        self.X_cv_norm = (X_cv - self.miu) / np.sqrt(self.var + 1e-8)

        self.y_train = np.array([self.convert2_OneHotEncodeing(y) for y in y_train])
        self.y_cv = np.array([self.convert2_OneHotEncodeing(y) for y in y_cv])
        self.y_test = np.array([self.convert2_OneHotEncodeing(y) for y in y_test])

        self.layers = layers
        self.Weights = []
        self.Gamma = [np.random.randn(1, layer.neurons) * 0.01 for layer in layers]
        self.Biases = [0.0 for _ in layers]
        self.V_sigma = [0.0 for _ in layers]
        self.V_miu = [0.0 for _ in layers]
        self.V_dg = [0.0 for _ in layers]
        self.S_dg = [0.0 for _ in layers]
        self.V_dw = []
        self.S_dw = []
        self.V_db = []
        self.S_db = []

        self.j_train = np.array([])
        self.j_cv = np.array([])
        self.accuracy = np.array([])
        self.accuracy_cv = np.array([])
        
        for index, layer in enumerate(self.layers):
            if index == 0:
                self.Weights.append(np.random.randn(
                    self.X_train_norm.shape[1], layer.neurons) * 0.01)
            else:
                self.Weights.append(np.random.randn(
                    self.layers[index - 1].neurons, layer.neurons) * 0.01)
            self.V_dw.append(np.zeros_like(self.Weights[index]))
            self.V_db.append(np.zeros_like(self.Biases[index]))
            self.S_dw.append(np.zeros_like(self.Weights[index]))
            self.S_db.append(np.zeros_like(self.Biases[index]))

    def convert2_OneHotEncodeing(self, index):
        e = np.zeros((10))
        e[index] = 1.0
        return e

    def data_normalize(self, X):
        miu = np.mean(X, axis=0, keepdims=True)
        var = np.var(X, axis=0, keepdims=True)
        X_norm = (X - miu) / np.sqrt(var + 1e-8)
        return X_norm, miu, var

    def parameters(self):
        n_weights = 0
        for W, B in zip(self.Weights, self.Biases):
            n_weights += np.size(W) + np.size(B)
        return n_weights

    def dense(self, A_in, W, b, activation, gamma, index, beta, testing=False):
        Z = np.matmul(A_in, W)
        Z_norm, miu, var = self.data_normalize(Z)

        if testing:
            Z_norm = (Z - self.V_miu[index]) / (self.V_sigma[index] + self.epsilon)
        else: 
            self.V_miu[index] = beta * self.V_miu[index] - (1.0 - beta) * miu
            self.V_sigma[index]  = beta * self.V_sigma[index] - (1.0 - beta) * np.sqrt(var)

        self.std.append(np.sqrt(var + self.epsilon))
        Z_tilde = Z_norm * gamma + b
        self.Z.append(Z)
        self.Z_norm.append(Z_norm)
        A_out = activation(Z_tilde)
        return A_out

    def forwardPropagation(self, X, beta=0.99, testing=False):
        self.A = [X]
        self.Z = []
        self.Z_norm = []
        self.std = []
        for index, (layer, A, W, b, G) in enumerate(zip(self.layers, self.A, self.Weights, self.Biases, self.Gamma)):
            self.A.append(self.dense(A, W, b, layer.activation, G, index, beta, testing))
        self.y_predict = self.A[-1]

    def backwardPropagation(self, y_batch, learning_rate, lambda_, beta1, beta2, t):
        delta = self.cross_entropy_grad(y_batch) / y_batch.shape[0]
        for index in reversed(np.arange(len(self.layers))):
            db = np.sum(delta, axis=0, keepdims=True)
            dg = np.sum(self.Z_norm[index] * delta, axis=0, keepdims=True)
            delta *= self.Gamma[index] / self.std[index]
            dw = np.matmul(self.A[index].T, delta)

            self.V_dw[index], self.S_dw[index] = beta1 * self.V_dw[index] + (1.0 - beta1) * dw, beta2 * self.S_dw[index] + (1.0 - beta2) * (dw ** 2)
            self.V_db[index], self.S_db[index] = beta1 * self.V_db[index] + (1.0 - beta1) * db, beta2 * self.S_db[index] + (1.0 - beta2) * (db ** 2)
            self.V_dg[index], self.S_dg[index] = beta1 * self.V_dg[index] + (1.0 - beta1) * dg, beta2 * self.S_dg[index] + (1.0 - beta2) * (dg ** 2)
            
            V_dw_correct, S_dw_correct = self.V_dw[index] / (1.0 - beta1 ** (t + 1)), self.S_dw[index] / (1.0 - beta2 ** (t + 1))
            V_db_correct, S_db_correct = self.V_db[index] / (1.0 - beta1 ** (t + 1)), self.S_db[index] / (1.0 - beta2 ** (t + 1))
            V_dg_correct, S_dg_correct = self.V_dg[index] / (1.0 - beta1 ** (t + 1)), self.S_dg[index] / (1.0 - beta2 ** (t + 1))

            adam_dw = learning_rate * V_dw_correct / (np.sqrt(S_dw_correct) + self.epsilon)
            adam_db = learning_rate * V_db_correct / (np.sqrt(S_db_correct) + self.epsilon)
            adam_dg = learning_rate * V_dg_correct / (np.sqrt(S_dg_correct) + self.epsilon)

            self.Weights[index] = self.Weights[index] * (1 - learning_rate * lambda_ / y_batch.shape[0]) - adam_dw
            self.Biases[index] -= adam_db
            self.Gamma[index] -= adam_dg

            if index == 0: break
            delta = np.matmul(
                delta, self.Weights[index].T) * self.layers[index - 1].activation_drev(self.Z[index - 1])

    def fit(self, epochs=100, learning_rate=0.02, beta1=0.9, beta2=0.999, beta3 = 0.99, lambda_=0.0, mini_batch=128, learning_rate_decay=0.1):
        num_batches = self.m // mini_batch
        tic = time()

        self.forwardPropagation(self.X_cv_norm, beta3, True)
        self.j_cv = np.append(self.j_cv, self.cross_entropy(self.y_cv))
        acc = 100.0 * self.getAccuracy(y=self.y_cv)
        self.accuracy_cv = np.append(self.accuracy_cv, acc)

        self.forwardPropagation(self.X_train_norm, beta3)
        self.j_train = np.append(self.j_train, self.cross_entropy(self.y_train))
        acc = 100.0 * self.getAccuracy(y=self.y_train)
        self.accuracy = np.append(self.accuracy, acc)

        print(f'epoch: {0}')
        print(f'training set prediction accuracy {acc} ,, training set cost {self.j_train[0]}')
        print(f'learning rate: {learning_rate}')
        
        for epoch in np.arange(1, epochs + 1):
            start, end = 0, 0

            for batch in np.arange(num_batches):
                end = (batch + 1) * mini_batch
                X_batch, y_batch = self.X_train_norm[start:end, :], self.y_train[start:end, :]
                start = end
                self.forwardPropagation(X_batch, beta3)
                self.backwardPropagation(y_batch, learning_rate, lambda_, beta1, beta2, batch)

            X_batch, y_batch = self.X_train_norm[end::, :], self.y_train[end::, :]
            self.forwardPropagation(X_batch, beta3)
            self.backwardPropagation(y_batch, learning_rate, lambda_, beta1, beta2, batch)

            self.forwardPropagation(self.X_cv_norm, beta3)
            self.j_cv = np.append(self.j_cv, self.cross_entropy(self.y_cv))
            acc = 100.0 * self.getAccuracy(y=self.y_cv)
            self.accuracy_cv = np.append(self.accuracy_cv, acc)

            self.forwardPropagation(self.X_train_norm, beta3)
            self.j_train = np.append(self.j_train, self.cross_entropy(self.y_train))
            acc = 100.0 * self.getAccuracy(y=self.y_train)
            self.accuracy = np.append(self.accuracy, acc)

            print(f'learning rate: {learning_rate}')
            if np.abs(self.j_train[epoch - 1] - self.j_train[epoch]) <= 1e-5: learning_rate *= (1.0 + learning_rate_decay * 5)
            else: learning_rate /= (1.0 + learning_rate_decay * (epoch - 1))

            if epoch % 10 == 0:
                print(f'epoch: {epoch}')
                print(f'training set prediction accuracy {acc} ,, training set cost {self.j_train[epoch]}')

        toc = time()
        self.executed_time = toc - tic
        self.testModel()

    def getAccuracy(self, y):
        return np.sum(
            np.argmax(self.y_predict, axis=1) == np.argmax(y, axis=1)
            ) / self.y_predict.shape[0]

    def testModel(self):
        x = np.arange(len(self.j_train))
        self.j_fig = go.Figure(go.Scatter(x=x, y=self.j_train, name='j_train'))

        self.j_fig.add_trace(go.Scatter(x=x, y=self.j_cv, name='j_cv'))

        self.j_fig.update_layout( 
            title='Cost function of training & cross validating examples', 
            xaxis_title='epochs', 
            yaxis_title='J_train & J_cv', 
            template='plotly_dark').show()

        self.accuracy_fig = go.Figure(go.Scatter(x=x, y=self.accuracy, name='accuracy_train'))

        self.accuracy_fig.add_trace(go.Scatter(x=x, y=self.accuracy_cv, name='accuracy_cv'))

        self.accuracy_fig.update_layout( 
            title='Accuracy of training examples', 
            xaxis_title='epochs', 
            yaxis_title='accuracy', 
            template='plotly_dark').show()

    def saveModel(self, index, learning_rate=0.02, beta1=0.9, beta2=0.999, lambda_=0.0, mini_batch=128, learning_rate_decay=0.1):
        if not isdir(f'./models/model_{index}/'):
            mkdir(f'./models/model_{index}/')

        # saving the plots we have done through the cost functions plots
        self.j_fig.write_image(f'./models/model_{index}/j_graph.png')
        self.accuracy_fig.write_image(f'./models/model_{index}/accuracy_graph.png')

        # saving parameters
        for i, (w, b) in enumerate(zip(self.Weights, self.Biases)):
            np.savetxt(f'./models/model_{index}/W_{i}.txt', w, fmt='%1.9f')
            np.savetxt(f'./models/model_{index}/B_{i}.txt', b, fmt='%1.9f')
        
        # saving the cost functions arrays
        np.savetxt(f'./models/model_{index}/J-cv.txt', self.j_cv, fmt='%1.9f')
        np.savetxt(f'./models/model_{index}/J-train.txt', self.j_train, fmt='%1.9f')
        np.savetxt(f'./models/model_{index}/accuracy.txt', self.accuracy, fmt='%1.9f')
        np.savetxt(f'./models/model_{index}/accuracy_cv.txt', self.accuracy_cv, fmt='%1.9f')
        
        # add model's summary 
        with open(f'./models/model_{index}/model_summary.md', "w") as f:
            f.write(f'# ***Model {index}***\n\n')
            f.write(f'Here is the summary of a trained model for the MNIST dataset.\n\n')
            f.write(f'## **1. Model Design**\n')
            f.write(f'## This model is consisted of *{len(self.layers)}* layers\n\n')

            for i in np.arange(len(self.layers)):
                f.write(f'Layer {i + 1}:\n')
                f.write(f'Layer {i + 1} is consisted of *{self.Weights[i].shape[1]}* neurons.\n\n')
                f.write(f'so the shape of its *Weights and Biases* are:\n\n')
                f.write(f'- Weights = {self.Weights[i].shape}\n\n')
                f.write(f'- Biases = {self.Biases[i].shape}\n\n')

            f.write(f'The total parameters of this model = {self.parameters()}\n')

            f.write(f'## **3. Model\'s Hyperparametes**\n')

            f.write(
                f'''- ### Model\'s hyperparameters are:\n
                Batch size (mini batch): {mini_batch} training examples\n
                Learning rate (alpha): {learning_rate}\n
                Learning rate decay: {learning_rate_decay}\n
                Regularization term -L2 regularization- (lambda): {lambda_}\n
                Gradient descent with momentum hyperparameter (beta 1): {beta1}\n
                RMSprop hyperparameter (beta 2): {beta2}\n\n'''
            )
            f.write(f'## **3. Model\'s Accuracy**\n')

            self.forwardPropagation(self.X_train_norm)
            train_loss = self.cross_entropy(self.y_train)
            f.write(f'- ### Model\'s accuracy of the training examples:\
                {self.getAccuracy(self.y_train) * 100}\n\n')
            
            f.write(f'![Accuracy of training examples](./accuracy_graph.png)\n\n')

            self.forwardPropagation(self.X_cv_norm)
            cv_loss = self.cross_entropy(self.y_cv)
            f.write(f'- ### Model\'s accuracy of the cross validating examples:\
                {self.getAccuracy(self.y_cv) * 100}\n\n')
            
            self.forwardPropagation(self.X_test_norm)
            test_loss = self.cross_entropy(self.y_test)
            f.write(f'- ### Model\'s accuracy of the testing examples:\
                {self.getAccuracy(self.y_test) * 100}\n\n')

            f.write(f'## **4. Model\'s Losses**\n')

            f.write(f'- ### Model\'s losses of the training examples: {train_loss}\n\n')

            f.write(f'![Cost function of training examples](./j_graph.png)\n\n')

            f.write(f'- ### Model\'s losses of the cross validating examples: {cv_loss}\n\n')

            f.write(f'- ### Model\'s losses of the testing examples: {test_loss}\n\n')

            f.write(f'## **5. Model\'s Executed time**\n')

            f.write(f'- ### The executed time: {int(self.executed_time / 60)} minutes,, \
            {int(self.executed_time % 60)} seconds ,, \
                {((self.executed_time - int(self.executed_time)) * 1000):.2f} milli seconds, along {len(self.accuracy) - 1} epochs\n\n')

    def loadModel(self, index):
        path = f'./models/model_{index}/'
        files = [f for f in listdir(path) if isfile(join(path, f))]
        self.Biases, self.Weights = [], []

        for file in files:
            for i in np.arange(len(files)):
                if file == f'B_{i}.txt':
                    self.Biases.append(np.loadtxt(join(path, f'B_{i}.txt')).reshape(1, -1))
                    break
                elif file == f'W_{i}.txt':
                    self.Weights.append(np.loadtxt(join(path, f'W_{i}.txt')))
                    break

        if len(self.Biases) != len(self.Weights) != len(self.layers):
            print('failed to load the model due to an error:')
            print('len(self.Biases) not equal len(self.Weights)')
            print('please make sure you are loading the right model')
            self.Biases, self.Weights = [], []
            return

        if isfile(join(path, 'J-train.txt')):
            self.j_train = np.loadtxt(join(path, 'J-train.txt'))
        else : print('failed to load J-train file')

        if isfile(join(path, 'accuracy.txt')):
            self.j_train = np.loadtxt(join(path, 'accuracy.txt'))
        else : print('failed to load accuracy file')

        if isfile(join(path, 'J-cv.txt')):
            self.j_train = np.loadtxt(join(path, 'J-cv.txt'))
        else : print('failed to load J-cv file')

        if isfile(join(path, 'accuracy_cv.txt')):
            self.j_train = np.loadtxt(join(path, 'accuracy_cv.txt'))
        else : print('failed to load accuracy_cv file')

        for w, b in zip(self.Weights, self.Biases):
            print(f'Weights: {w.shape}')
            print(f'Biases: {b.shape}')

    def cross_entropy(self, y):
        y_pred_clipped = np.clip(self.y_predict, self.epsilon, 1 - self.epsilon)
        return np.mean(-y * np.log(y_pred_clipped))

    def cross_entropy_grad(self, y): return self.y_predict - y