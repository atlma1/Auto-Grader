from keras.datasets import mnist
(train_X, train_y), (test_X, test_y) = mnist.load_data()
from matplotlib import pyplot
'''for i in range(9):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(train_X[i], cmap=pyplot.get_cmap('gray'))
    pyplot.show()'''
print(train_X[1])

