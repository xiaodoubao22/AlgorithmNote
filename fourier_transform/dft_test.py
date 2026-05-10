import numpy as np
import matplotlib.pyplot as plt

REAL = 0
IMAG = 1

N = 16
K = N
dispK = -1
IFFTpreK = int(0.5* N) # 0.0~0.5

def FuncSin(w:float, phi:float, t:np.array) -> np.array:
    return np.sin(w * t + phi)

def FuncSquare(w:float, phi:float, t:np. array) -> np.array:
    return np.sign(np.sin(w * (t + 0.5)) + phi)

def DFTBaseFunctions(w:float, t:np.array) -> np.ndarray:
    # 基函数根据欧拉公式展开 eps(-i * (2 * pi * k * n) / N) = cos(2 * pi * k * n / N) - i * sin(2 * pi * k * n / N)
    # base也称为DFT矩阵
    base = np.zeros((2, K, N))
    for k in range(0, K):
        base[REAL][k] = np.cos(w * k * t)
        base[IMAG][k] = -np.sin(w * k * t)
    return base

def DFT(y:np.array, base:np.ndarray) -> np.array:
    # F(k) = Σn:[0~N-1] f(x) * eps(-i * (2 *pi * k * n / N))
    # 其中 eps(-i * (2 * pi * k * n / N)) = base[REAL][k][n] + base[IMAG][k][n] * i
    # 矩阵形式 F = DFTMatrix @ y.T
    ft = np.zeros((2, K))
    ft[REAL] = base[REAL] @ y.T
    ft[IMAG] = base[IMAG] @ y.T
    return ft
 
def IDFT(w:float, preK:int, ft:np.array) -> np.array:
    # f(k) = (1/N) * Σk:[0~N-1] F(x) * eps(i * (2 * pi * k * n))
    # 记F(x) = a + b * i
    # F(x) * eps(i * (pi_2 * k * n))
    # = (a + b * i)(cos(wkn) + sin(wkn) * i)
    # = (a * cos(wkn) - b * sin(wkn) + (a * sin(wkn) + b * cos(wkn)) * i)
    # = (a * cos(wkn) - b * sin(wkn)) (虚部共轭对称抵消)
    # 矩阵形式 y = IDFTMatrix @ F.T
    k = np.linspace(0, N - 1, N)
    ibase = np.zeros((2, N, K))
    for n in range(0, N):
        ibase[REAL][n] = np.cos(w * k * n)
        ibase[IMAG][n] = np.sin(w * k * n)
    ift = ibase[REAL] @ ft[REAL].T - ibase[IMAG] @ ft[IMAG].T
    return ift

if __name__ == '__main__':
    t = np.linspace(0, N - 1, N)
    w = (2 * np.pi / N)
    #y = FuncSin(w, phi=0, t=t)     # 正弦函数信号
    y = FuncSquare(w=w, phi=0, t=t) # 方波信号

    # DFT
    base = DFTBaseFunctions(w, t)
    ft = DFT(y, base)
    absFFT = (np.abs(ft[REAL]) ** 2 + np.abs(ft[IMAG]) ** 2) ** 0.5

    # IDFT
    ift = IDFT(w, IFFTpreK, ft)

    # 显示图像
    fig = plt.figure(figsize=(12,6))

    # 原信号
    ax1=fig.add_subplot(2, 2, 1)
    ax1.set_title(f"origin signal", loc="center", fontsize=10)
    ax1.set_xlim(-1, N)
    ax1.plot(t, y, color='black', marker='o', markersize=6, linestyle='')
    if dispK >= 0:
        ax1.plot(t, base[REAL][dispK], color='red', marker='o', markersize=3, linestyle='')
        ax1.plot(t, base[IMAG][dispK], color='blue', marker='o', markersize=3, linestyle='')
    ax1.grid()

    # 频谱图复数
    ax2=fig.add_subplot(2, 2, 2)
    ax2.set_title(f"DFT(complex)", loc="center", fontsize=10)
    ax2.set_xlim(-1, N)
    ax2.set_ylim(np.min(ft) * 1.1, np.max(ft) * 1.1)
    if dispK >= 0:
        ax2.vlines(t[dispK], 0, ft[REAL][dispK], color='red')
        ax2.plot(t[dispK], ft[REAL][dispK], color='red', marker='o', markersize=6, linestyle='')
        ax2.vlines(t[dispK], 0, ft[IMAG][dispK], color='blue')
        ax2.plot(t[dispK], ft[IMAG][dispK], color='blue', marker='o', markersize=5, linestyle='')
    else:
        ax2.vlines(t, 0, ft[REAL], color='red')
        ax2.plot(t, ft[REAL], color='red', marker='o', markersize=6, linestyle='')
        ax2.vlines(t, 0, ft[IMAG], color='blue')
        ax2.plot(t, ft[IMAG], color='blue', marker='o', markersize=5, linestyle='')
    ax2.grid()

    # 频谱图绝对值
    ax4=fig.add_subplot(2, 2, 4)
    ax4.set_title(f"DFT(abs)", loc="center", fontsize=10)
    ax4.set_xlim(-1, N)
    ax4.set_ylim(-np.max(absFFT) * 0.1, np.max(absFFT) * 1.1)
    ax4.vlines(t, 0, absFFT, color='black')
    ax4.plot(t, absFFT, color='black', marker='o', markersize=5, linestyle='')
    ax4.grid()

    # IDFT
    ax3=fig.add_subplot(2, 2, 3)
    ax3.set_title(f"IDFT", loc="center", fontsize=10)
    ax3.plot(t, ift, color='black', marker='', markersize=5, linestyle='-')
    ax3.grid()

    plt.show()

