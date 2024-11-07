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
#print(train_set_y)

print("Shape of train_set_x:", train_set_x.shape)
print("Shape of train_set_y:", train_set_y.shape)
print("Shape of test_set_x:", test_set_x.shape)
print("Shape of test_set_y:", test_set_y.shape)
print("Pixel range:", train_set_x.min(), train_set_x.max())

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

def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0.0
    return w, b

def propagate(w, b, X, Y, i):
    m = X.shape[1]

    A = sigmoid(np.dot(w.T,X)+b)
    
    if i == 0:
        print("shape of w", w.shape)
        print("shape of X", X.shape)
        print('shape of A', A.shape)
        print('A:', A)
        print('Y:', Y)

    cost = (-1/m)*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))
    print('cost', cost)

    dw = (1/m)*np.dot(X,(A-Y).T)
    db = (1/m)*np.sum(A-Y)

    cost = np.squeeze(np.array(cost))

    grads = {"dw": dw,
             "db": db}
    
    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost):
    print('number of iteration:', num_iterations)
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)
    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y, i)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate*dw
        b = b - learning_rate*db

        if i % 10 == 0:
            costs.append(cost)
#            print('costs_length',len(costs),costs[len(costs)-1],cost)

            if print_cost:
                print("Cost after iteration %i: %f: %f" %(i, cost, cost-costs[len(costs)-2]))

    params = {"w": w,
              "b": b}
    grads = {"dw": dw,
             "db": db}

    return params, grads, costs

def predict(w, b, X):
    m = X.shape[1]
    print("predict:shape of X", X.shape)
    print("predict:shape of m", m)
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    print("predict:shape of w", w.shape)  

    A = sigmoid(np.dot(w.T, X) + b)
    print("predict:shape of A", A.shape)  
    for i in range (A.shape[1]):
        if A[0, i]> 0.5:
            Y_prediction[0, i] = 1

        else:
            Y_prediction[0, i] = 0

    return Y_prediction
def draw(costs):
    #plot learning curve
    costs = np.squeeze(logistic_regression_model['costs'])
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundred)')
    plt.title("Learning rate =" + str(logistic_regression_model["learning_rate"]))
    plt.show()



def model (X_train, Y_train, X_test, Y_test, num_iterations, learning_rate, print_cost = True):
    w, b = initialize_with_zeros(X_train.shape[0])
    print('shape of w:', w.shape)

    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    w = params["w"]
    b = params["b"]
    Y_prediction_test = predict (w, b, X_test)
    Y_prediction_train = predict (w, b, X_train)
    print("Y_pred_test",Y_prediction_test)
    print("Y_pred_train",Y_prediction_train)

    if print_cost:
        print("train accuracy: {} %", format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %", format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d

logistic_regression_model = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 3, learning_rate=0.005)

learning_rates = [0.01, 0.001, 0.0001]
model = {}

for lr in learning_rates:
    print ("Training a model with learning rate: " + str(lr))
    models[str(lr)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=1500, learning_rate=lr, print_cost=False)
    print ('\n' + "-------------------------------------------------------" + '\n')

for lr in learning_rates:
    plt.plot(np.squeeze(models[str(lr)]["costs"]), label=str(models[str(lr)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations (hundreds)')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()

# change this to the name of your image file
my_image = "my_image.jpg"   

# We preprocess the image to fit your algorithm.
fname = "images/" + my_image
image = np.array(Image.open(fname).resize((num_px, num_px)))
plt.imshow(image)
image = image / 255.
image = image.reshape((1, num_px * num_px * 3)).T
my_predicted_image = predict(logistic_regression_model["w"], logistic_regression_model["b"], image)

print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
