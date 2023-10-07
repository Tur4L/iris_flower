#this function is for reading the data giving the x and y datesets
import numpy as np

def read_data(file_name = "iris.data"):
    x_data = np.zeros((150,4))
    y_data = np.zeros(150)
    with open("iris.data","r") as f:
        for idline, line in enumerate(f):
            index = 0
            data = []
            if line != "\n":
                for i in line.split()[0]:
                    if i == ",":
                        element = "".join(data)
                        data = []
                        x_data[idline,index] = float(element)
                        index += 1
                    else:
                        data.append(i)

                    if "".join(data) == "Iris-":
                        element = line.split()[0][21:]
                        if element == "setosa":
                            iris = 0
                        elif element == "versicolor":
                            iris = 1
                        elif element == "virginica":
                            iris = 2
                            
                        y_data[idline] = iris
                        data = []
    return np.array(x_data), np.array(y_data.astype(int))