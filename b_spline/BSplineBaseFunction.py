import matplotlib.pyplot as plt
import numpy as np

EPS = 1e-6

knots = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
#knots = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0])
controlPoints = np.array([[0.0, 1.0], [1.0, -1.0], [2.0, 1.0], [3.0, -1.0], [4.0, 1.0],
                          [5.0, -1.0], [6.0, 1.0], [7.0, -1.0], [8.0, 1.0], [9.0, -1.0]])

def B1(k:int, u:float) -> float:
    if k < 0 or k > len(knots) - 1:
        return 0.0  # 不合法
    if u >= knots[k] and u < knots[k + 1]:
        return 1.0
    return 0.0

def BasisFunctionTraversal(k:int, d:int, u:float) -> float:
    if d == 1:
        return B1(k, u)
    # k1
    denom = (knots[k + d - 1] - knots[k])
    nom = (u - knots[k])
    k1 = 0.0
    if denom > EPS:
        k1 = nom / denom
    
    # k2
    denom = (knots[k + d] - knots[k + 1])
    nom = (knots[k + d] - u)
    k2 = 0.0
    if denom > EPS:
        k2 = nom / denom

    res = BasisFunctionTraversal(k, d - 1, u) * k1 + BasisFunctionTraversal(k + 1, d - 1, u) * k2
    return res

def BasisFunction(k:int, d:int, u:float):
    if k < 0 or k > len(knots) - d:
        return 0.0  # 不合法
    return BasisFunctionTraversal(k, d, u) 

if __name__ == '__main__':
    uArr = np.arange(0.0 + EPS, 5.0 - EPS, 0.01)
    b1Arr = np.zeros([len(knots), len(uArr)])
    b2Arr = np.zeros([len(knots), len(uArr)])
    b3Arr = np.zeros([len(knots), len(uArr)])
    b4Arr = np.zeros([len(knots), len(uArr)])
    pointsX = np.zeros(len(uArr))
    pointsY = np.zeros(len(uArr))

    for k in range(len(knots) - 1):
        for i in range(len(uArr)):
            b1Arr[k, i] = BasisFunction(k, 1, uArr[i])

    for k in range(len(knots) - 2):
        for i in range(len(uArr)):
            b2Arr[k, i] = BasisFunction(k, 2, uArr[i])

    for k in range(len(knots) - 3):
        for i in range(len(uArr)):
            b3Arr[k, i] = BasisFunction(k, 3, uArr[i])

    for k in range(len(knots) - 4):
        for i in range(len(uArr)):
            b4Arr[k, i] = BasisFunction(k, 4, uArr[i])

    b = 3
    for i in range(len(uArr)):
        point = np.array([0.0, 0.0])
        for k in range(len(knots) - b):
            point += controlPoints[k] * BasisFunction(k, b, uArr[i])
            print(uArr[i], controlPoints[k], BasisFunction(k, b, uArr[i]))
        pointsX[i] = point[0]
        pointsY[i] = point[1]

    fig = plt.figure(figsize=(6,8))
    ax1=fig.add_subplot(4, 1, 1)
    ax2=fig.add_subplot(4, 1, 2)
    ax3=fig.add_subplot(4, 1, 3)
    ax4=fig.add_subplot(4, 1, 4)
    for k in range(len(knots) - 1):
        ax1.plot(uArr, b1Arr[k,:], color='blue')
    for k in range(len(knots) - 2):
        ax2.plot(uArr, b2Arr[k,:], color='red')
    for k in range(len(knots) - 3):
        ax3.plot(uArr, b3Arr[k,:], color='orange')
    ax4.plot(pointsX, pointsY, color='black')
    ax4.scatter(controlPoints[:, 0], controlPoints[:, 1], color='black')
    plt.show() 

