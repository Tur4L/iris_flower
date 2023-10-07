#here we test the differences between the 3 algorithms

import nn_adam
import nn_rmsprop
import softmax
import matplotlib.pyplot as plt

"""In this file, we will compare the machine learning algorithms"""

accuracy_adam = []
accuracy_rmsprop = []
accuracy_softmax = []

adam = nn_adam.nn_adam()
rmsprop = nn_rmsprop.nn_rmsprop()
soft = softmax.softmax()

for i in range(20):
    accuracy_adam.append(adam.main())
    accuracy_rmsprop.append(rmsprop.main())
    accuracy_softmax.append(soft.main())

adam_average = sum(accuracy_adam)/20
rmsprop_average = sum(accuracy_rmsprop)/20
softmax_average = sum(accuracy_softmax)/20

print("Adam: ",adam_average)
print("Rmsprop: ",rmsprop_average)
print("Softmax: ",softmax_average)

plt.figure()
plt.plot(accuracy_adam)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig("q2_adam_accs")

plt.figure()
plt.plot(accuracy_rmsprop)
plt.xlabel("Runs")
plt.ylabel("RMSprop-Accuracy")
plt.savefig("q2_rmsprop_accs")

plt.figure()
plt.plot(accuracy_softmax)
plt.xlabel("Runs")
plt.ylabel("Softmax-Accuracy")
plt.savefig("q2_softmax_accs")