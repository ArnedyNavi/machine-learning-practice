import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

def trainingSet():
    f = open("ex1data2.txt", "r")
    data = []
    x = np.array([0,0])
    y = np.array([0])

    for i in f:
        data.append(i)

    for i in range(len(data)):
        p = data[i].split(",")
        x = np.vstack([x, [float(p[0]), float(p[1])]])
        y = np.vstack([y, float(p[2])])
    x = np.delete(x, 0, 0)
    y = np.delete(y, 0, 0)
    return x, y

def normalization(x):
    size = np.shape(x)
    m = size[0]
    n = size[1]

    data_mean = np.mean(x, axis = 0)
    data_std = np.std(x, axis = 0)
    for i in range(n):
        mean = data_mean[i]
        std = data_std[i]
        x[:, i] = (x[:, i] - mean) / std
    return [x, data_mean, data_std]

def addx0(x):
    size = np.shape(x)
    m = size[0]
    n = size[1]

    x0 = np.array([1])
    for i in range(m - 1):
        x0 = np.vstack([x0, [1]])
    x = np.hstack([x0, x])
    return x

def countCost(theta, x, y):
    h = np.matmul(theta, np.transpose(x))

    dif = np.subtract(h, y)
    m = np.shape(x)[1]
    J = 1/(2*m) * np.sum(np.square(dif))
    return J

def countDerivative(theta, x, y):
    h = np.matmul(theta, np.transpose(x))

    dif = np.subtract(h, np.transpose(y))
    
    m = np.shape(x)[0]

    deriv = 1/m * np.matmul(np.transpose(x), np.transpose(dif))
    
    deriv = np.transpose(deriv)
    return deriv

def costFunction(theta, x, y):
    J = countCost(theta, x, y)
    grad = countDerivative(theta, x, y)  
    return J, grad

def gradientDescent(x, y, alpha, iter):
    size = np.shape(x)
    m = size[0]
    n = size[1]

    theta = np.array([])
    for i in range(n):
        theta = np.append(theta, [0])

    all_cost = np.array([])
    for i in range(iter):
        cost = countCost(theta, x, y)
        all_cost = np.append(all_cost, cost)
        # print(f"Iteration {i + 1} - J = {cost}")
        theta = theta - alpha * countDerivative(theta, x, y)
        
    return theta, all_cost


if __name__ == "__main__":
    data = trainingSet()
    x_bef = data[0]
    x = data[0]
    [x, mean, std] = normalization(data[0])
    norm = True
    y = data[1]
    x = addx0(x)

    size = np.shape(x)
    m = size[0]
    n = size[1]

    init_theta = np.array([])
    for i in range(n):
        init_theta = np.append(init_theta, [0])

    # Alpha = 1,2 for normalized, 0,00000043 for unnormalized
    if norm == False:
        res = gradientDescent(x,y,0.00000043, 1000)
    else:
        res = gradientDescent(x,y,1.2, 1000)
    res_1 = minimize(countCost, x0=init_theta, args=(x, y), method='SLSQP')

    theta = res[0][0]
    theta_1 = res_1.x
    [cost, grad] = costFunction(theta, x, y)
    [cost_1, grad_1] = costFunction(theta_1, x,y)

    print(f"Optimum Theta (Gradient Descent) = {np.round(theta, decimals=2)}")
    print(f"Optimum Theta (Minimize scipy)= {np.round(theta_1, decimals=2)}")
    print(f"Cost at optimum Theta 1= {cost}")
    print(f"Cost at optimum Theta 2= {cost_1}")
    
    
    x1_data = np.arange(1000, 5000, 10)
    x2_data = np.arange(0, 10, 1)
    y_data = np.matmul(theta, np.transpose(x))


    X, Y = np.meshgrid(x1_data, x2_data)
    if norm == True:
        Z = theta[0] + theta[1] * ((X - mean[0])/std[0]) + theta[2] * ((Y - mean[1])/std[1])
    else:
        Z = theta[0] + theta[1] * X + theta[2] * Y

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')

    x_data = trainingSet()[0]
    x_val = x_data[:,0]
    y_val = x_data[:,1]
    z_val = y

    ax.scatter3D(x_val, y_val, z_val, color="blue")
    plt.show()
    

    