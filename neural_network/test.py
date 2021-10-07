from neural_net import *
import numpy as np
import scipy.io as sio
import scipy.optimize as opt
import pandas as pd
import json
from PIL import Image

def getTheta():
    f = open("theta.txt", "r")
    data = json.loads(f.read())
    f.close()
    return data

if __name__ == "__main__":
    theta = getTheta()
    theta1 = np.asarray(theta["theta1"])
    theta2 = np.asarray(theta["theta2"])

    data = trainingSet()
    x_data = data[0]
    x = addBias(x_data)
    y_data = data[1]
    k = np.amax(y_data)
    y = classMapping(y_data, k)

    [m, n] = np.shape(x)
    L = 3
    s = np.array([n-1, 25, k])

    theta = rollTheta(theta1, theta2)
    displayData(theta1[:,1:], 100, "s")
    displayData(x_data, 400, "r")
    accuracy = predictAccuracy(theta, x, y_data, s)
    print(accuracy[0])
    
    






