import numpy as np
import scipy.io as sio
import scipy.optimize as opt
import pandas as pd
import json
from PIL import Image

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def trainingSet():
    mat_contents = sio.loadmat("ex3data1.mat")
    x = mat_contents['X']
    y = mat_contents['y']
    return x, y

def trainedTheta():
    mat_contents = sio.loadmat("ex3weights.mat")
    theta1 = mat_contents['Theta1']
    theta2 = mat_contents['Theta2']
    return theta1, theta2

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

def randomInitialized(L, s):
    init_e = 0.12
    theta = []
    for i in range(L - 1):
        theta_temp = np.random.rand(s[i + 1], s[i] + 1) * 2 * init_e - init_e
        theta.append(theta_temp)
    return theta

def rollTheta(theta1, theta2):
    theta1 = theta1.flatten()
    theta2 = theta2.flatten()
    theta = np.concatenate([theta1, theta2])
    return theta

def unrollTheta(theta, s):
    input_size = s[0] + 1
    hidden_size = s[1]
    output_size = s[2]
    n = input_size * hidden_size
    theta1 = theta[0:n]    
    theta2 = theta[n:]
    theta1 = theta1.reshape(hidden_size, input_size)
    theta2 = theta2.reshape(output_size, hidden_size + 1)
    return theta1, theta2

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
        indices = np.append(indices, indices)
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
    
def sigmoid(z):
    den = 1 + np.exp(-1 * z)
    h = 1/den
    return h

def forwardPropagation(theta, x):
    [m, n] = np.shape(x)
    theta1 = theta[0]
    theta2 = theta[1]

    # Forward Propagation
    z_2 = np.matmul(theta1, np.transpose(x))
    a_2_non = np.transpose(sigmoid(z_2))
    a_2 = addBias(a_2_non)
    z_3 = np.matmul(theta2, np.transpose(a_2))
    a_3 = np.transpose(sigmoid(z_3))
    return [z_2, a_2, z_3, a_3]

def costFunction(init_theta, x, y, l, s):
    theta = unrollTheta(init_theta, s)
    res = forwardPropagation(theta, x) 
    
    [m, n] = np.shape(x)
    h = res[3]
    theta1 = theta[0]
    theta2 = theta[1]
    
    J_unreg = 0
    y = np.transpose(y)
    for j in range(k):
        J_unreg = J_unreg + (-1/m * (np.matmul(y[j,:], np.log(h[:,j])) + np.matmul((1 - y[j]), np.log(1-h[:,j]))))
    
    theta1_reg = theta1[:,2:]
    theta2_reg = theta2[:,2:]

    reg = l/(2*m) * (np.sum(theta1_reg[2:].flatten() ** 2) + np.sum(theta2_reg[2:].flatten() ** 2))
    J = J_unreg + reg
    return J

def sigmoidGradient(z):
    g = sigmoid(z) * (1-sigmoid(z))
    return g

def gradient(theta, x, y, l, s):
    theta = unrollTheta(theta, s)
    res = forwardPropagation(theta, x) 
    k = np.shape(y)[1]
    [m, n] = np.shape(x)

    h = res[3]
    a_1 = x
    z_2 = res[0]
    a_2 = res[1]
    z_3 = res[2]
    
    theta1 = theta[0]
    theta2 = theta[1]

    delta3 = np.zeros((m, k))
    for i in range(k):
        delta3[:,i] = h[:,i] - y[:,i]
    
    delta2_ = np.matmul(np.transpose(theta2), np.transpose(delta3)) * np.transpose(sigmoidGradient(addBias(np.transpose(z_2))))
    delta2 = delta2_[1:, :]

    Delta2_unreg = np.matmul(np.transpose(delta3), a_2)/m
    Delta1_unreg = np.matmul(delta2, a_1)/m

    Delta1 = Delta1_unreg + (l/m) * theta1
    Delta2 = Delta2_unreg + (l/m) * theta2
    Delta1[:, 1] = Delta1_unreg[:, 1]
    Delta2[:, 1] = Delta2_unreg[:, 1]

    grad = rollTheta(Delta1, Delta2)
    return grad

def predictAccuracy(theta, x, y, s):
    [m, n] = np.shape(x)
    theta = unrollTheta(theta, s)
    h = forwardPropagation(theta, x)[3]
    defect = []
    wrong = []
    correct = 0
    for i in range(m):
        if np.argmax(h[i]) + 1 == y[i]:
            correct = correct + 1
        else:
            defect.append(i)
            wrong.append(np.argmax(h[i]) + 1)
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
    L = 3
    s = np.array([n-1, 25, k])

    init_theta = randomInitialized(L, s)
    l = 1
    init_theta = rollTheta(init_theta[0], init_theta[1])

    res = opt.minimize(fun=costFunction, x0=init_theta, args=(x, y, l, s), method='TNC', jac=gradient, options={"disp" : False, "maxiter" : 1000})
    theta_temp = res.x
    theta = unrollTheta(theta_temp, s)
    theta1 = theta[0]
    theta2 = theta[1]

    result = dict({"theta1" : theta1, "theta2" : theta2})
    f = open("theta2.txt", "w")

    f.write(json.dumps(result, cls=NumpyEncoder))
    f.close()

    #[theta1, theta2] = trainedTheta()
    print(theta)
    #theta_temp = rollTheta(theta1, theta2)
    
    acc = predictAccuracy(theta_temp, x, y_data, s)

    accuracy = acc[0]
    print(f"Learning Accuracy = {accuracy}%")
