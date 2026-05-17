import numpy as np
import matplotlib.pyplot as plt
import time

REAL = 0
IMAG = 1

LOG_2_N = 10
N = 2 ** LOG_2_N
K = N
dispK = -1
IFFTpreK = int(0.5* N) # 0.0~0.5

print("N=",N)

def FuncSin(w:float, phi:float, t:np.array) -> np.array:
    return np.sin(w * t + phi)

def FuncSquare(w:float, phi:float, t:np. array) -> np.array:
    return np.sign(np.sin(w * (t + 0.5)) + phi)

def DFTBaseFunctions(w:float, t:np.array) -> np.ndarray:
    # 基函数根据欧拉公式展开 exp(-i * (2 * pi * k * n) / N) = cos(2 * pi * k * n / N) - i * sin(2 * pi * k * n / N)
    # base也称为DFT矩阵
    tk = t.reshape(N, 1) @ t.reshape(1, N)
    base = np.zeros((2, K, N))
    base[REAL] = np.cos(w * tk)
    base[IMAG] = -np.sin(w * tk)
    return base

def DFT(y:np.array, base:np.ndarray) -> np.array:
    # F(k) = Σn:[0~N-1] f(x) * exp(-i * (2 *pi * k * n / N))
    # 其中 exp(-i * (2 * pi * k * n / N)) = base[REAL][k][n] + base[IMAG][k][n] * i
    # 矩阵形式 F = DFTMatrix @ y.T
    ft = np.zeros((2, K))
    ft[REAL] = base[REAL] @ y.T
    ft[IMAG] = base[IMAG] @ y.T
    return ft
 
def IDFT(w:float, preK:int, ft:np.array) -> np.array:
    # f(k) = (1/N) * Σk:[0~N-1] F(x) * exp(i * (2 * pi * k * n))
    # 记F(x) = a + b * i
    # F(x) * exp(i * (pi_2 * k * n))
    # = (a + b * i)(cos(wkn) + sin(wkn) * i)
    # = (a * cos(wkn) - b * sin(wkn) + (a * sin(wkn) + b * cos(wkn)) * i)
    # = (a * cos(wkn) - b * sin(wkn)) (虚部共轭对称抵消)
    # 矩阵形式 y = IDFTMatrix @ F.T
    k = np.linspace(0, N - 1, N)
    kn = k.reshape(N, 1) @ k.reshape(1, N)

    ibase = np.zeros((2, N, K))
    ibase[REAL] = np.cos(w * kn)
    ibase[IMAG] = np.sin(w * kn)
    ift = ibase[REAL] @ ft[REAL].T - ibase[IMAG] @ ft[IMAG].T
    return ift

# 递归法
cosTerm, sinTerm = np.array([0]), np.array([0])
P = np.array([0])
def FFTRecur(st:int, n:int, step:int) -> np.array:
    global P
    if n == 1:
        return np.array([[P[st]], [0]])
    n_2 = n // 2

    # Pe, Po = P[0::2], P[1::2]
    ye, yo = FFTRecur(st, n_2, step * 2), FFTRecur(st + step, n_2, step * 2)

    # wj = exp(-i * (2  *pi * j / N)) = cos(2 * pi * j / N) - i * sin(2 * pi * j / N)
    global cosTerm, sinTerm

    # wj * yo
    oddTerm = np.array([
        cosTerm[::step] * yo[REAL] + sinTerm[::step] * yo[IMAG],
        cosTerm[::step] * yo[IMAG] - sinTerm[::step] * yo[REAL]
    ])

    y = np.zeros((2, n)) # 2:实部+虚部
    y[:, :n_2] = ye + oddTerm
    y[:, n_2:] = ye - oddTerm
    return y

def FFT(y:np.array):
    global P
    P = y
    n = len(P)
    n_2 = n // 2

    # wj = exp(-i * (2  *pi * j / N))
    # wjExp = (2 * pi * j / N)
    wjExp = (2.0 * np.pi  / n) * np.linspace(0, n_2 - 1, n_2)

    # wj = cos(2 * pi * j / N) - i * sin(2 * pi * j / N)
    global cosTerm, sinTerm
    cosTerm, sinTerm = np.cos(wjExp), np.sin(wjExp)
    return FFTRecur(0, n, 1)

# 迭代法
def FFTIter(y:np.array):
    n = len(y)
    n_2 = n // 2

    # 生成倒序二进制  0 4 2 6  1 5 3 7... ...
    forwardBinary = np.linspace(0, n - 1, n, dtype=int)
    backwardBinary = np.zeros_like(forwardBinary)
    for _ in range(LOG_2_N):
        backwardBinary = backwardBinary << 1        # 左移腾出位置
        backwardBinary |= forwardBinary & 1         # 取原数最低位，加到反转数
        forwardBinary = forwardBinary >> 1          # 原数右移一位

    # 对输入采样点重新排序
    P = np.array([y[i] for i in backwardBinary])
    
    # 计算j次单位根
    # wj = cos(2 * pi * j / N) - i * sin(2 * pi * j / N)
    wjExp = (2.0 * np.pi  / n) * np.linspace(0, n_2 - 1, n_2)
    cosTerm, sinTerm = np.cos(wjExp), np.sin(wjExp)

    yPre = np.array([P, np.zeros(n)])
    yPre = np.concatenate([yPre[REAL].reshape(-1, 1, 1), yPre[IMAG].reshape(-1, 1, 1)], axis=1)
    yCur = np.zeros((2, n)) # 2:实部+虚部
    for i in range(LOG_2_N):
        ye = yPre[0::2]
        yo = yPre[1::2]

        step = 2 ** (LOG_2_N - 1 - i)

        oddTermReal = cosTerm[::step] * yo[:, REAL:REAL+1:, :] + sinTerm[::step] * yo[:, IMAG:IMAG+1:, :]
        oddTermImag = cosTerm[::step] * yo[:, IMAG:IMAG+1:, :] - sinTerm[::step] * yo[:, REAL:REAL+1:, :]
        oddTerm = np.concatenate([oddTermReal, oddTermImag], axis=1)

        yCur = np.concatenate([ye + oddTerm, ye - oddTerm], axis=2)
        yCur, yPre = yPre, yCur
    return yPre[0]

if __name__ == '__main__':
    t = np.linspace(0, N - 1, N)
    w = (2 * np.pi / N)
    #y = FuncSin(w, phi=0, t=t)     # 正弦函数信号
    y = FuncSquare(w=w, phi=0, t=t) # 方波信号

    # DFT
    start = time.perf_counter()
    base = DFTBaseFunctions(w, t)
    ft = DFT(y, base)
    end = time.perf_counter()
    print(f"dft time      :{end - start:.6f} s")

    # FFT 递归法
    start = time.perf_counter()
    fft = FFT(y)
    end = time.perf_counter()
    print(f"fft time      :{end - start:.6f} s")

    # 迭代法
    start = time.perf_counter()
    fftiter = FFTIter(y)
    end = time.perf_counter()
    print(f"fft iter time :{end - start:.6f} s")

    # numpy 内置算法
    start = time.perf_counter()
    fftNumpy = np.fft.fft(y)
    end = time.perf_counter()
    print(f"fft np time   :{end - start:.6f} s")

    # 选择最终结果
    ft = fftiter
    #ft = np.array([fftNumpy.real, fftNumpy.imag])
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

