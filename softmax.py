#implementing softmax multiclassification
import numpy as np
from read_data import read_data
import matplotlib.pyplot as plt

class softmax():
    def __init__(self):
        
        self.x_data, self.y_data = read_data()
        _random_indices = np.arange(len(self.x_data))
        np.random.shuffle(_random_indices)

        self.x_data = self.x_data[_random_indices]
        self.y_data = self.y_data[_random_indices]

        #hyperparameters
        self.alpha = 0.8
        self.N_class = 3
        self.batch_size = 10
        self.Max_epoch = 50
        self.decay = 0.

        #creating the data
        self.train_data_x = self.x_data[:90]
        self.train_data_y = self.y_data[:90]

        self.val_data_x = self.x_data[90:120]
        self.val_data_y = self.y_data[90:120]

        self.test_data_x = self.x_data[120:]
        self.test_data_y = self.y_data[120:]

        #some variables for simplicity
        self.d= self.train_data_x.shape[1]
        self.N_train = self.train_data_x.shape[0]
        self.N_val = self.val_data_x.shape[0]
        self.W = None

        #train variables:
        self.best_epoch = None
        self.best_acc = None
        self.W_best = None

    def softmax(self,z):
        exp_z = np.exp(z - np.max(z,axis=1,keepdims=True))
        return exp_z / np.sum(exp_z,axis=1,keepdims=True)
    
    def predict(self,X,W,t=None):
        z = np.dot(X,W)
        y = self.softmax(z)
        t_hat = np.argmax(y,axis=1)

        are_same = 0
        for index, e in enumerate(t_hat):
            if e == t[index]:
                are_same += 1

        acc = are_same/len(t)
        loss = -np.sum(np.log(y[np.arange(len(y)), t.flatten()])) / float(len(t))

        return y,t_hat, loss, acc
    
    def train(self,X_train,y_train,X_val,y_val):
        train_losses = []
        valid_accs = []
        self.best_epoch = 0
        self.best_acc = 0
        self.best_W = np.random.randn(self.d, self.N_class) / np.sqrt(self.d)
        self.W = np.random.randn(self.d, self.N_class) / np.sqrt(self.d)

        for epoch in range(self.Max_epoch):
            random_indices = np.arange(self.N_train)
            np.random.shuffle(random_indices)
            X_train = X_train[random_indices]
            y_train = y_train[random_indices]

            for batch_start in range(0,self.N_train,self.batch_size):
                batch_end = batch_start + self.batch_size
                X_batch = X_train[batch_start:batch_end]
                Y_batch = y_train[batch_start:batch_end]

                y_pred,_,loss,_ = self.predict(X_batch,self.W,Y_batch)
                train_losses.append(loss)

                delta = y_pred - np.eye(self.N_class)[Y_batch.flatten()]
                grad = np.dot(X_batch.T,delta)

                self.W -= self.alpha*grad

            _,_,_,acc = self.predict(X_val,self.W,y_val)
            valid_accs.append(acc)

            if acc > self.best_acc:
                self.best_acc = acc
                self.best_epoch = epoch
                self.W_best = np.copy(self.W)

        return train_losses, valid_accs
    
    def main(self):
        train_losses, valid_accs = self.train(self.train_data_x,self.train_data_y,self.val_data_x,self.val_data_y)
        _,_,_,acc_test = self.predict(self.test_data_x,self.W_best,self.test_data_y)

        return acc_test

if __name__ == "__main__":
    s = softmax()
    s.main()