import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import h5py
import copy

train_dataset = h5py.File('/home/sourav/Desktop/logistic_regression/dataset/train_catvnoncat.h5',"r")
test_dataset = h5py.File('/home/sourav/Desktop/logistic_regression/dataset/test_catvnoncat.h5',"r")
print("Train dataset keys:", list(train_dataset.keys()))
print("Test dataset keys:", list(test_dataset.keys()))

# Access specific data in the dataset
train_set_x = train_dataset["train_set_x"][:]  # Training images
train_set_y = train_dataset["train_set_y"][:]  # Training labels

test_set_x = test_dataset["test_set_x"][:]  # Test images
test_set_y = test_dataset["test_set_y"][:]  # Test labels
#train_set_x_orig, train_set_y, test_set_y_orig, test_set_y, classes = load_dataset()
print("Shape of train_set_x:", train_set_x.shape)
print("Shape of train_set_y:", train_set_y.shape)
print("Shape of test_set_x:", test_set_x.shape)
print("Shape of test_set_y:", test_set_y.shape)
print("Pixel range:", train_set_x.min(), train_set_x.max())
#index=13
#plt.imshow(train_set_x[index])
#print(train_set_y[index])
#for i in range(train_set_x.shape[0]):
#    print (i,train_set_y[i])
#plt.savefig("sample_image.png")

train_set_y=train_set_y.reshape(1,209)
test_set_y=test_set_y.reshape(1,50)
train_set_x_flatten = train_set_x.reshape(train_set_x.shape[0],-1).T
test_set_x_flatten = test_set_x.reshape(test_set_x.shape[0],-1).T
print(train_set_x_flatten.shape)
print(test_set_x_flatten.shape)
print(train_set_y.shape)
print(test_set_y.shape)

# normalization of the data set
train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255

def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s

print(sigmoid(np.array([0, 2])))
print(sigmoid(np.array([-0.1, 0, 0.1])))

def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0.0
    return w, b

w, b = initialize_with_zeros(12288)
print(w, b)

def propagate(w, b, X, Y):
    m = X.shape[1]

    A = sigmoid(np.dot(w.T,X)+b)
    cost = (-1/m)*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))

    dw = (1/m)*np.dot(X,(A-Y).T)
    db = (1/m)*np.sum(A-Y)

    cost = np.squeeze(np.array(cost))

    grads = {"dw": dw,
             "db": db}
    
    return grads, cost

# test
w = np.array([[1], [2]])
b = 1.5
X = np.array([[1, -2, -1], [3, 0.5, -3.2]])
Y = np.array([[1, 1, 0]])

grads, cost = propagate(w, b, X, Y)

print (grads, cost)
print ("dw=" + str(grads["dw"]))
print ("db=" + str(grads["db"]))
print ("cost =" + str(cost))

# end test


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost):
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)
    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate*dw
        b = b - learning_rate*db

        if i % 100 == 0:
            costs.append(cost)

            if print_cost:
                print("Cost after iteration %i: %f" %(i, cost))

    params = {"w": w,
              "b": b}
    grads = {"dw": dw,
             "db": db}

    return params, grads, costs

# test
params, grads, costs= optimize(w, b, X, Y, 300, 0.09,True)

print ("w = ", params["w"])
print ("b = ", params["b"])
print ("dw = ", grads["dw"])
print ("db = ", grads["db"])
print ("Cost ", costs)
# end test

def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) + b)
    for i in range (A.shape[1]):
        if A[0, i]> 0.5:
            Y_prediction[0, i] = 1

        else:
            Y_prediction[0, i] = 0

    return Y_prediction

# test
w = np.array([[0.1124579],[0.23106775]])
b = -0.3
X = np.array([[1., -1.1, -3.2], [1.2, 2.0, 0.1]])
print ("prediction =", predict(w, b, X))

def model (X_train, Y_train, X_test, Y_test, num_iteration=2000, learning_rate = 0.5, print_cost = False):
    w, b = initialize_with_zeros(X_train.shape[0])
    w = params["w"]
    b = params["b"]
    Y_prediction_test = predict (w, b, X_test)
    Y_prediction_train = predict (w, b, X_train)

    if print_cost:
        print("train accuracy: {} %", format(100 - np.mean(np.abs(Y-prediction_train - Y_train)) * 100))
        print("test accuracy: {} %", format(100 - np.mean(np.abs(Y-prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b.
         "learning_rate": learning_rate,
         "num_iteration": num_iteration}

    return d

logistic_regression_model = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate=0.005)

