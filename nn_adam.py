#implementing neural network with Adam optimizer
import numpy as np
from read_data import read_data
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

class nn_adam():
    def __init__(self):
         
        self.x_data, self.y_data = read_data()

        self.train_data_x = self.x_data[:120]
        self.train_data_y = self.y_data[:120]
        self.test_data_x = self.x_data[120:]
        self.test_data_y = self.y_data[120:]

        self.model = None

    def train(self,epoch=66):

        self.model= Sequential()
        self.model.add(Dense(100,input_shape=(4,), activation="relu"))
        self.model.add(Dense(100, activation="relu"))
        self.model.add(Dense(3, activation='softmax'))
        self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        self.model.fit(self.train_data_x,self.train_data_y, epochs=epoch)

        """To save a model"""
        #self.model.save("nn_adam.model")

    def main(self):

        """if we need to creata a new model, uncomment below"""
        self.train()

        """Tuning for number of epochs"""
        # best_accuracy = 0
        # best_epoch = 0
        # epoches = list(range(1,101))
        # for epoch in epoches:
        #     self.train(epoch)
        #     _,accuracy = self.model.evaluate(self.test_data_x,self.test_data_y)
        #     if accuracy > best_accuracy:
        #         best_accuracy = accuracy
        #         best_epoch = epoch
        # print("Best epoch: ",best_epoch)
        # print("Accuracy: ",best_accuracy)

        """If using existing model, uncomment below"""
        #self.model = tf.keras.models.load_model("nn_adam.model")

        _,accuracy = self.model.evaluate(self.test_data_x,self.test_data_y)

        return accuracy
if __name__ == "__main__":
    nn = nn_adam()
    nn.main()

