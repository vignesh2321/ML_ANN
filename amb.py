import numpy as np
import matplotlib.pyplot as plt
import glob as gb
import random
import os
import cv2
dataset_folder ="./Ambulance Dataset/"
code = {"ambulance" : 1 , "non-ambulance" : 0}
def place(n):
  for item in code:
    if n == code[item]:
      return item

train = dataset_folder + "train/"
test = dataset_folder + "test/"

image_resize = 128

train_data = []
xtrain = []
ytrain = []

for folder in os.listdir(train):
  files = gb.glob(pathname=str(train + folder + "/*.jpg"))
  for file in files:
    train_image=plt.imread(file)
    image_size=cv2.resize(train_image,(image_resize,image_resize))
    train_data.append([np.array(image_size), code[folder]])

print("Train Data Size: ", len(train_data))

random.shuffle(train_data)
for td in train_data:
  xtrain.append(td[0])
  ytrain.append(td[1])


test_data = []
xtest = []
ytest = []

# Testing images
for folder in os.listdir(test):
  files = gb.glob(pathname=str(test + folder + "/*.jpg"))
  for file in files:
    test_image=plt.imread(file)
    image_size=cv2.resize(test_image,(image_resize,image_resize))
    test_data.append([np.array(image_size), code[folder]])

print("Test Data Size: ", len(test_data))

# Shuffle the testing images
random.shuffle(test_data)
for td in test_data:
  xtest.append(td[0])
  ytest.append(td[1])


x_train = np.array(xtrain)
y_train = np.array(ytrain)
x_test = np.array(xtest)
y_test = np.array(ytest)

code = dict([(value, key) for key, value in code.items()])
plt.figure(figsize=(20,20))
for i in range(18):
  plt.subplot(6,6,i+1)
  plt.axis("off")
  plt.title(code[y_train[i]])
  plt.imshow(x_train[i])

x_train = x_train.reshape(x_train.shape[0], -1, 1)
x_test = x_test.reshape(x_test.shape[0], -1, 1)
y_train = y_train.reshape((y_train.shape[0], 1))
y_test  = y_test .reshape((y_test .shape[0], 1))

x_train = x_train/255
x_test = x_test/255

print("Training data x-values:", x_train.shape)
print("Training data y-values:", y_train.shape)
print("Testing data x-values:", x_test.shape)
print("Testing data y-values:",y_test.shape)


def initialize_layers(layers):
  # np.random.seed(0)
  # Store the weights and biases in dictionary
  WB = {}

  for l in range(1, len(layers)):
    # Weight dim = [current layer dim, prev layer dim]
    WB["W" + str(l)] = np.random.randn(layers[l], layers[l - 1]) * 0.01  # np.full((layers[l], layers[l-1]), 0.1)
    print("W" + str(l), WB["W" + str(l)].shape)

    # Weight gradients initialized to zero (for backprop)
    WB["dW" + str(l)] = np.zeros((layers[l], layers[l - 1]))
    print("dW" + str(l), WB["dW" + str(l)].shape)

    # Bias dim = [current layer dim]
    WB["b" + str(l)] = np.ones((layers[l], 1)) * 0.01  # np.full(layers[l], 0.1)
    print("b" + str(l), WB["b" + str(l)].shape)

    WB["db" + str(l)] = np.zeros((layers[l], 1))
    print("db" + str(l), WB["db" + str(l)].shape)

  return WB

def sigmoid(net):
  '''
  Calculate sigmoid activation value
  '''
  return 1/(1+np.exp(-net))


def linear_activation_forward(X, W, b):
  '''
  Calculate net value
  Calculate the sigmoid activation value for net

  History - Stores the inputs, Weights and biases
  '''
  net = np.dot(W, X) + b
  output = sigmoid(net)
  history = (X, W, b)

  return output, history


def model_linear_activation_forward(WB, X, layers):
  '''
  For an training example
    store the history
    Calculate it's output
  '''
  output = X
  length = len(layers)

  for l in range(1, length):
    output_prev = output
    W = WB["W" + str(l)]
    b = WB["b" + str(l)]
    WB["X" + str(l)] = output_prev
    output, history = linear_activation_forward(output_prev, W, b)

  return output, WB

def error_output_unit(T, O):
  '''
  T - Target Values
  O - Output Values
  '''
  E = O * (1 - O) * (T - O)

  return E


def error_hidden_unit(W, E, O):
  '''
  W - weight to the next layer
  E - Error calculated for the next layer
  O - Output of the current layer
  '''
  W = W.T
  S = np.sum(np.dot(W, E))
  E = O * (1 - O) * S

  return E


def weight_update(W, b, E, X, LR, alpha, prevDW, prevDb):
  '''
  W - original weights
  b - biases
  E - Error values
  X - Input values
  LR - Learning Rate

  dW - weight updates
  db - bias updates

  AW - altered weights
  Ab - altered biases

  alpha - momentum (if 0 -> no momentum)
  prevDW - dW of previous iteration
  prevDb - db of previous iteration
  '''
  dW = LR * np.dot(E, X.T)
  db = LR * E * 1

  AW = W + dW + (alpha * prevDW)
  Ab = b + db + (alpha * prevDb)

  return AW, dW, Ab, db

def back_prop_output_unit(T, O, W, b, X, LR, alpha, prevDW, prevDb):
  E = error_output_unit(T, O)
  AW, dW, Ab, db = weight_update(W, b, E, X, LR, alpha, prevDW, prevDb)

  return E, AW, dW, Ab, db

def back_prop_hidden_unit(Wnext, Enext, O, W, b, X, LR, alpha, prevDW, prevDb):
  E = error_hidden_unit(Wnext, Enext, O)
  AW, dW, Ab, db = weight_update(W, b, E, X, LR, alpha, prevDW, prevDb)

  return E, AW, dW, Ab, db


def model_back_prop(WB, O, T, LR, layers, alpha):
  length = len(layers) - 1
  W = WB["W" + str(length)]
  X = WB["X" + str(length)]
  b = WB["b" + str(length)]
  prevDW = WB["dW" + str(length)]
  prevDb = WB["db" + str(length)]

  E, W, dW, b, db = back_prop_output_unit(T, O, W, b, X, LR, alpha, prevDW, prevDb)

  # Update dW, db Error (for Adding momentum)
  WB["dW" + str(length)] = dW
  WB["E" + str(length)] = E
  WB["db" + str(length)] = db

  # Update the weights, biases (for Adding momentum)
  WB["W" + str(length)] = W
  WB["b" + str(length)] = b

  for l in range(length - 1, 0, -1):
    # Next layers's weight and error (previously calculated)
    Wnext = W
    Enext = E

    # Current layer's output (i.e) next layers input
    O = X

    # Current layer's weights and biases
    W = WB["W" + str(l)]
    b = WB["b" + str(l)]

    # Input to current layer
    X = WB["X" + str(l)]
    prevDW = WB["dW" + str(l)]
    prevDb = WB["db" + str(l)]

    # Update weights
    E, W, dW, Ab, db = back_prop_hidden_unit(Wnext, Enext, O, W, b, X, LR, alpha, prevDW, prevDb)

    # Update dW, db Error (for Adding momentum)
    WB["dW" + str(l)] = dW
    WB["E" + str(l)] = E
    WB["db" + str(l)] = db

    # Update the weights, biases (for Adding momentum)
    WB["W" + str(l)] = W
    WB["b" + str(l)] = b
  return WB


def train_model(WB, x_train, y_train, num_epochs, LR, layers, alpha, history):
  total_ex = len(x_train)

  # Run the model for the given number of epochs
  for e in range(num_epochs):
    print("Epoch ", e + 1)
    error = 0
    accuracy = 0
    for i in range(total_ex):
      # Forward Propagation
      output, WB = model_linear_activation_forward(WB, x_train[i], layers)
      # Backward Propagation
      WB = model_back_prop(WB, output, y_train[i], LR, layers, alpha)
      # Accuracy and Error
      error += (y_train[i] - output) ** 2
      if (output >= 0.5 and y_train[i] == 1) or (output < 0.5 and y_train[i] == 0):
        accuracy += 1

    # Calculate the final accuracy and error
    final_error = np.squeeze(error) / total_ex
    history["error"].append(final_error)
    final_accuracy = accuracy / total_ex
    history["accuracy"].append(final_accuracy)
    print("Error: ", final_error, "\tAccuracy: ", final_accuracy)

    # To avoid overfitting
    if final_accuracy >= 0.99:
      print("99.5% Accuracy reached. Stopping the training")
      break


def model_predict(WB, X, layers):
  output = X

  for l in range(1, len(layers)):
    output_prev = output
    W = WB["W" + str(l)]
    b = WB["b" + str(l)]
    output, history = linear_activation_forward(output_prev, W, b)

  return output

def test_model(WB, x_test, y_test, layers):
  total_ex = len(x_test)
  error = 0
  accuracy = 0
  # Predict for all the test examples
  for i in range(total_ex):
    output = model_predict(WB, x_test[i], layers)
    #print("Predicted Output: ", np.squeeze(output),  "\tTest Output: ", np.squeeze(y_test[i]))

    # Calculate the error and accuracy
    error += (y_test[i] - output) ** 2
    if (output >= 0.5 and y_test[i] == 1) or (output < 0.5 and y_test[i] == 0):
      accuracy += 1

  # Calculate the final accuracy and error
  final_error = np.squeeze(error)/total_ex
  final_accuracy = accuracy/total_ex

  print("Error: ", final_error, "\tAccuracy: ", final_accuracy)

def plot_graphs(history, string):
  plt.plot(history[string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string])
  plt.show()

input_layer = x_train.shape[1]
hidden_1 = 32
output_layer = 1
layers = [input_layer, hidden_1, output_layer]
WB = initialize_layers(layers)
history = {"accuracy" : [],
           "error" : []}

num_epochs = 500
LR = 0.005
alpha = 0.3

train_model(WB, x_train, y_train, num_epochs, LR, layers, alpha, history)

# With momentum test accuracy
test_model(WB, x_test, y_test, layers)

plot_graphs(history, "accuracy")
plot_graphs(history, "error")

#print(WB)

#fn=input("enter name")
#fn=fn+".jpg"
result = dict()


def test1():
  fn = "ambu.jpg"
  result = dict()
  image = cv2.imread(fn)
  image_np = cv2.resize(image, (image_resize, image_resize))

  image_np = image_np.reshape(-1, 1) / 255
  output = model_predict(WB, image_np, layers)
  print("\n--------------------------")
  print("Predicted Output: ", np.squeeze(output))
  if output >= 0.35:
    print(fn + " is an ambulance\n")
    result[fn] = "ambulance"
  else:
    print(fn + " is not an ambulance\n")
    result[fn] = "non-ambulance"





test1()



import pickle

with open("p.pkl", "wb") as f:
  pickle.dump("amb.py", f)
