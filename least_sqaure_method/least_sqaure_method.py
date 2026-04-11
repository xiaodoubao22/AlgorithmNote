import numpy as np
import matplotlib.pyplot as plt

## 多项式拟合
def LSMPoly(inputX, inputY, dMax):
    # 模型: y = c[0] + c[1] * x + c[2] * x^2 + ... + c[n] * x^n
    A = np.array([(inputX ** i) for i in range(dMax + 1)]).T
    b = np.array([inputY]).T

    # 求解 A * x = b
    # 法线方程: (A' * A) * x_ = A' * b
    # 求解: x_ = inv(A' * A) * (A' * b)
    AA = A.T @ A
    if np.linalg.det(AA) < 1e-5:
        print('det zero')
        return np.array([0 for i in range(0, dMax + 1)])
    res = np.linalg.inv(AA) @ (A.T @ b)
    return np.flip(res.T[0])   # 结果顺序从高次到低次 c[n], c[n-1], ... , c[1], c[0]


# 计算误差
def RMSEPloy(inputX, inputY, p):
    # np.polyval(p, x): 求多项式 res = p[0]*x**(N-1) + p[1]*x**(N-2) + ... + p[N-2]*x + p[N-1]
    predictY = np.polyval(p, inputX)
    return np.sqrt(np.mean((predictY - inputY) ** 2))

def RMSEExp(inputX, inputY, c):
    predictY = c[0] * np.exp(c[1] * inputX)
    return np.sqrt(np.mean((predictY - inputY) ** 2))


# 显示
def DisplayPoly(ax, vecPolyCoef, start, end, step):
    dispX = np.arange(start, end, step)
    for p in vecPolyCoef:
        dispY = np.polyval(p, dispX)
        ax.plot(dispX, dispY)
        ax.annotate('aaaa',
                    xy=(dispX[-1], dispY[-1]),
                    xytext=(dispX[-1], dispY[-1]),
                    arrowprops=dict(facecolor='red', shrink=0.005))

def DisplayExp(ax, vecExpCoef, start, end, step):
    dispX = np.arange(start, end, step)
    for c in vecExpCoef:
        dispY = c[0] * np.exp(c[1] * dispX)
        ax.plot(dispX, dispY)


if __name__ == '__main__':
    # 输入数据
    # inputX = np.array([-1.0, 0.0, 1.0, 2.0])
    # inputY = np.array([1.0, 0.0, 0.0, -2.0])
    # dataRange = [-1.5, 2.5, 0.1]

    inputX = np.array([1950.0, 1955.0, 1960.0, 1965.0, 1970.0, 1975.0, 1980.0])
    inputY = np.array([ 53.05,  73.04,  98.31, 139.78, 193.48, 260.20, 320.39])
    dataRange = [1940.0, 1990, 1]

    # 一次多项式拟合模型
    res1 = LSMPoly(inputX, inputY, 1)
    print("res1=", res1)
    rmse1 = RMSEPloy(inputX, inputY, res1)
    print("rmse1=", rmse1)
    print("")

    # 二次多项式拟合模型
    res2 = LSMPoly(inputX, inputY, 2)
    print("res2=", res2)
    rmse2 = RMSEPloy(inputX, inputY, res2)
    print("rmse2=", rmse2)
    print("")

    # 指数模型  y = c1 * e ^ (c2 * t)
    # 转化 ln(y) = ln(c1) + c2 * t; k = ln(c1)
    lnInputY = np.log(inputY)
    resExp = LSMPoly(inputX, lnInputY, 1)
    print("resExp=", resExp)
    rmseExp = RMSEPloy(inputX, lnInputY, resExp)
    print("rmseExp in log space=", rmseExp)
    resExp[1] = np.exp(resExp[1])   # c1 = exp(k)
    rmseExp = RMSEExp(inputX, inputY, np.flip(resExp))
    print("rmseExp in linear space=", rmseExp)
    print("")

    # 显示
    fig = plt.figure(figsize=(6,6))
    ax1=fig.add_subplot(1, 1, 1)
    ax1.scatter(inputX, inputY, color='black')
    DisplayPoly(ax1, [res1, res2], dataRange[0], dataRange[1], dataRange[2])
    DisplayExp(ax1, [np.flip(resExp)], dataRange[0], dataRange[1], dataRange[2])
    plt.show()