import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt  
import math

def trainingSet():
    f = open("ex2data1.txt", "r")
    x = np.array([0, 0])
    y = np.array([0])
    
    for j in f:
        data = []
        i = j
        data = i.split(",")
        x_j = [float(data[0]), float(data[1])]
        x = np.vstack([x, x_j])
        y = np.vstack([y, float(data[2])])
    x = np.delete(x, 0, axis=0)
    y = np.delete(y, 0, axis=0)
    return x, y

def trainingSet2():
    f = open("ex2data2.txt", "r")
    x = np.array([0, 0])
    y = np.array([0])
    
    for j in f:
        data = []
        i = j
        data = i.split(",")
        x_j = [float(data[0]), float(data[1])]
        x = np.vstack([x, x_j])
        y = np.vstack([y, float(data[2])])
    x = np.delete(x, 0, axis=0)
    y = np.delete(y, 0, axis=0)
    
    degree = 6
    x = featureMapping(x, 6)
    return x, y

def featureMapping(x, degree):
    m = np.shape(x)[0]

    x_mapped = np.zeros((m,1))

    for i in range(1, degree + 1):
        for j in range(i + 1):
            x_mapped = np.hstack([x_mapped, np.reshape(((x[:,0] ** (i - j)) * (x[:,1] ** j)), (m, 1))])
    x_mapped = np.delete(x_mapped, 0, axis=1)
    return x_mapped
            
def addBias(x):
    size = np.shape(x)
    m = size[0]
    n = size[1]

    x0 = np.array([[1]])
    for i in range(m - 1):
        x0 = np.vstack([x0, 1])
    x = np.hstack([x0, x])
    return x

def sigmoid(init_theta, x):
    m = np.shape(x)[0]
    theta = np.transpose(init_theta)

    den = (1.0 + np.exp(-1 * np.dot(theta, np.transpose(x))))
    h = 1.0/den
    h = h.reshape(m, 1)
    return h

def costFunction(init_theta, x, y, l):
    size = np.shape(x)
    m = size[0]
    n = size[1]
    
    theta = np.transpose(init_theta)
    
    h = sigmoid(theta, x)
    J = -1/m * ((np.dot(np.transpose(y), np.log(h))) + np.dot((1-np.transpose(y)), np.log(1-h))) + (l/(2*m) * (np.sum(theta.flatten()[1:] ** 2)))
    return J[0][0]

def gradient(init_theta, x, y, l):
    size = np.shape(x)
    m = size[0]
    n = size[1]

    theta = np.transpose(init_theta)
    h = sigmoid(theta, x)

    grad_unreg = (1./m * (np.dot(np.transpose(x), (h - y)))).flatten()
    grad = grad_unreg.flatten() + (l/m * theta.flatten())
    grad[0] = grad_unreg[0]
    return grad

def gradientDescent(theta, x, y, alpha, l, iter):
    size = np.shape(x)
    m = size[0]
    n = size[1]

    for i in range(iter):
        cost = costFunction(theta, x, y)
        grad = gradient(theta, x, y)
        #print(f"Iteration {i + 1} - J = {cost}")
        theta = theta - (alpha * grad)
    return theta

def predictAccuracy(theta, x, y):
    m = np.shape(x)[0]
    h = np.round(sigmoid(theta, x))

    diff = h - y
    correct = len(np.where(diff == 0)[0])
    
    accuracy = correct/m * 100
    return accuracy
    
if __name__ == "__main__":
    data = trainingSet2()
    x_data = data[0]
    x = addBias(x_data)
    y = data[1]
    init_theta = np.zeros(np.shape(x)[1])
    #Test init_theta
    #init_theta = np.array([-24, 0.2, 0.2])
    l = 1
    #result_tnc = opt.minimize(fun=costFunction, x0=init_theta, args=(x, y, l), method='tnc', jac=gradient)
    #print(result_tnc.x)
    result_bfgs = opt.minimize(fun=costFunction, x0=init_theta, args=(x, y, l), method='bfgs')
    print(result_bfgs.x)

    #theta = gradientDescent(init_theta, x, y, 0.0049480, 1, 10000)
    #print(theta)

    theta = result_bfgs.x
    print(f"Optimum Theta = {theta}")
    accuracy = predictAccuracy(theta, x, y)
    print(f"Accuracy = {accuracy}%")

    plt.figure()
    data = trainingSet2()
    x_data = data[0]
    y_data = data[1]
    
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]

    plt.scatter(x_data[pos, 0], x_data[pos, 1], marker='.', color='blue')
    plt.scatter(x_data[neg, 0], x_data[neg, 1], marker='x', color='red')

    if np.shape(x_data)[1] <= 2:
        X = np.arange(0, 1, 0.1)
        Y = (-1 * (theta[0] + theta[1] * X))/theta[2]
        plt.plot(X,Y)
    else:
        X = np.linspace(-1, 1.5, 50)
        Y = np.linspace(-1, 1.5, 50)

        Z = np.zeros((len(X), len(Y)))
        for i in range(len(X)):
            for j in range(len(Y)):
                featuredMap = featureMapping(np.reshape([X[i], Y[j]], (1, 2)), 6)
                feature = addBias(featuredMap)
                Z[i][j] = np.dot(feature, theta)
        
        Z = np.transpose(Z)
        plt.contour(X, Y, Z, 0, cmap='viridis')
    plt.show()







    
    

        
        
