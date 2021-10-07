import numpy as np
import scipy.io as sio
import scipy.optimize as opt
import pandas as pd
from PIL import Image

def trainingSet():
    mat_contents = sio.loadmat("ex3data1.mat")
    x = mat_contents['X']
    y = mat_contents['y']
    return x, y

def addBias(x):
    size = np.shape(x)
    m = size[0]
    n = size[1]

    x0 = np.array([[1]])
    for i in range(m - 1):
        x0 = np.vstack([x0, 1])
    x = np.hstack([x0, x])
    return x

def classMapping(y, k):
    size = np.shape(y)
    m = size[0]
    
    y_mapped = np.zeros((1, k))
    for i in range(m):
        y_arr = np.array([])
        for j in range(1, k + 1):
            y_temp = int(y[i] == j)
            y_arr = np.append(y_arr, y_temp)
        y_mapped = np.vstack([y_mapped, y_arr])
    y_mapped = np.delete(y_mapped, 0, axis=0)
    return y_mapped

#Display data, minimum size = 20
def displayData (x, n_image, method, **kwargs):
    data_image = np.zeros((n_image, n_image), dtype=np.uint8 )  
    size = np.shape(x)
    m = size[0]
    n = size[1]

    n_el = np.sqrt(n)
    elements = n_image/n_el

    num_elements = int(elements ** 2)
    if method == "r":
        indices = np.random.randint(0, (m-1), size=(2 * num_elements))
    elif method == "s":
        indices = range(0,m)
        indices = np.append(indices, indices)
    elif method == "g":
        indices = kwargs["defect"]
        indices = np.append(indices, indices)
        indices = np.append(indices, indices)

    w = 0
    i_ = 0
    for j in range(n_image):
        h = 0
        i = i_
        for k in range(n_image):
            data_image[j][k] = np.absolute(x[indices[i]][int(w + (h * n_el))]) / np.amax(np.absolute(x[indices[i]])) * 255
            if h >= (n_el - 1):
                h = 0
                i = i + 1
            else:
                h = h + 1
        if w >= (n_el - 1):
            w = 0
            i_ = int(i + elements)
        else: 
            w = w + 1
    
    image = Image.fromarray(data_image)
    image.show()         

def displayDefect(x, defect):
    total_def = len(defect)
    [m, n] = np.shape(x)

    n_image = np.sqrt(total_def*n)
    n_image = int(np.ceil(np.sqrt(n_image)) ** 2)
    displayData(x, n_image, "g", defect=defect)
    return 0
    
def sigmoid(init_theta, x):
    theta = init_theta.reshape(1, np.shape(init_theta)[0])
    theta = np.transpose(init_theta)

    den = 1 + np.exp(-1 * np.matmul(theta, np.transpose(x)))
    h = 1/den
    h = h.reshape(np.shape(x)[0], 1)
    return h

def predict(init_theta, x):
    theta = init_theta.reshape(1, np.shape(init_theta)[0])
    theta = np.transpose(init_theta)

    den = 1 + np.exp(-1 * np.matmul(theta, np.transpose(x)))
    h = 1/den
    return h

def costFunction(init_theta, x, y, l):
    [m, n] = np.shape(x)

    theta = np.transpose(init_theta)
    h = sigmoid(theta, x)
    
    J_unreg = -1/m * (np.matmul(y, np.log(h)) + np.matmul(1-y, np.log(1-h)))
    reg = l/(2*m) * np.sum(theta[2:].flatten() ** 2)
    J = J_unreg + reg
    return J

def gradient(theta, x, y, l):
    [m, n] = np.shape(x)
    h = sigmoid(theta, x)

    theta = np.transpose(init_theta)

    y = y.reshape(1, np.shape(y)[0])
    y = np.transpose(y)

    grad_unreg = (1/m * np.matmul(np.transpose(x), (h - y))).flatten()
    reg = l/m * theta.flatten()
    grad = grad_unreg + reg
    grad[0] = grad_unreg[0]

    return grad

def predictAccuracy(theta, x, y):
    k = np.shape(theta)[0]
    [m, n] = np.shape(x)
    correct = 0
    defect = []
    wrong = []
    y = y.flatten()
    for i in range(m):
        h_temp = np.array([])
        for j in range(k):
            h = predict(np.transpose(theta[j,:]), x[i,:])
            h_temp = np.append(h_temp, h)
        h_temp = h_temp
        if np.argmax(h_temp) + 1 == y[i]:
            correct = correct + 1
        else:
            defect.append(i)
            wrong.append(np.argmax(h_temp) + 1)
    accuracy = correct / m * 100
    return accuracy, defect, wrong

if __name__ == "__main__":
    data = trainingSet()
    x_data = data[0]
    x = addBias(x_data)
    y_data = data[1]
    k = np.amax(y_data)
    y = classMapping(y_data, k)

    [m, n] = np.shape(x)

    init_theta = np.zeros(n)
    theta = np.zeros((k, n))
    l = 10

    for i in range(k):
        res = opt.minimize(fun=costFunction, x0=init_theta, args=(x, y[:,i], l), method='TNC', jac=gradient)
        theta_temp = res.x
        theta[i, :] = theta_temp

    [accuracy, defect, wrong] = predictAccuracy(theta, x, y_data)
    print(f"Learning accuracy = {accuracy}")
    
    displayDefect(x_data, defect)
    f = open("defect.txt", "w")
    for i in range(len(defect)):
        y_def = y_data[defect[i]]
        y_wrong = wrong[i]
        f.write(f"{y_def} ---> {y_wrong} \n")
    f.close()
