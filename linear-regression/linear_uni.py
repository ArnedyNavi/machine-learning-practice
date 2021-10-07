import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

def trainingSet():
    f = open("ex1data1.txt", "r")
    data = []
    x = np.array([])
    y = np.array([])

    for i in f:
        data.append(i)

    for i in range(len(data)):
        p = data[i].split(",")
        x = np.append(x, [float(p[0])])
        y = np.append(y, [float(p[1])])
    return x, y

def addx0(x):
    x0 = np.array([])
    for i in range(len(x)):
        x0 = np.append(x0, [1])
    x = np.vstack([x0, x])
    return x

def costFunction(theta, x, y):
    h = np.matmul(theta, x)

    dif = np.subtract(h, y)
    m = np.shape(x)[1]
    J = 1/(2*m) * np.sum(np.square(dif))
    return J

def countDerivative(theta, x, y):
    h = np.matmul(theta, x)

    dif = np.subtract(h, y)

    m = np.shape(x)[1]
    deriv = 1/m * np.matmul(x, np.transpose(dif))
    return deriv

    
def gradientDescent(x, y, alpha, iter):
    theta = np.array([0,0])
    all_cost = np.array([])
    for i in range(iter):
        cost = costFunction(theta, x, y)
        all_cost = np.append(all_cost, cost)
        #print(f"Iteration {i + 1} - J = {cost}")
        theta = theta - alpha * countDerivative(theta, x, y)
    return theta, all_cost

if __name__ == "__main__":
    data = trainingSet()
    x = data[0]
    x = addx0(x)
    y = data[1]

    result = gradientDescent(x, y, 0.01, 3000)
    theta = result[0]
    costs = result[1]
    print(f"Optimum theta found = {theta}")

    plt.figure()
    x_predict = data[0]
    y_predict = theta[0] + theta[1] * data[0]
    plt.plot(x_predict,y,'x')
    plt.plot(x_predict, y_predict)

    plt.figure()
    iteration = np.arange(2,3000,1)
    plt.plot(iteration, costs[2:], color="red")

    plt.figure()
    theta_1 = np.arange(-10, 11, 1)
    theta_2 = np.arange(-10, 11, 1)

    X, Y = np.meshgrid(theta_1, theta_2)

    Z = np.zeros((21, 21))
    for i in range(21):
        for j in range(21):
            Z[i][j] = costFunction([X[i][j], Y[i][j]], x, y)

    ax = plt.axes(projection = '3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis')
    plt.show()



    
    
    

    