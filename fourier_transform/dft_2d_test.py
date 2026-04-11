import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

REAL = 0
IMAG = 1

FilterRad = 0.2
FilePath = "./data/Cameraman.bmp"

def ReadImage(filePath:str) -> np.ndarray:
    print(type(filePath))
    readImage = Image.open(filePath)
    print(f"格式: {readImage.format}, 大小: {readImage.size}, 模式: {readImage.mode}")
    return np.array(readImage) / 255.0

def AbsComplex(F:np.ndarray) -> np.ndarray:
    return (F[REAL] ** 2 + F[IMAG] ** 2) ** 0.5

def BaseImage(t:np.array) -> np.ndarray:
    N = len(t)
    K = N
    w = 2.0 * np.pi / N
    base = np.zeros((2, K, N))
    for k in range(0, K):
        base[REAL][k] = np.cos(w * t * k)
        base[IMAG][k] = -np.sin(w * t * k)
    return base

def DFT2D(baseY, baseX, oriImg):
    Y = oriImg.shape[0]
    X = oriImg.shape[1]
    U = X
    V = Y

    Fxv = np.zeros((2, V, X))
    for v in range(0, V):
        Fxv[REAL][v] = baseY[REAL][v] @ oriImg
        Fxv[IMAG][v] = baseY[IMAG][v] @ oriImg

    Fuv = np.zeros((2, V, U))
    for u in range(0, U):
        Fuv[REAL, :, u] = (Fxv[REAL] @ baseX[REAL][u].reshape(U, 1)).ravel() - \
            (Fxv[IMAG] @ baseX[IMAG][u].reshape(U, 1)).ravel()
        Fuv[IMAG, :, u] = (Fxv[REAL] @ baseX[IMAG][u].reshape(U, 1)).ravel() + \
            (Fxv[IMAG] @ baseX[REAL][u].reshape(U, 1)).ravel()
    return Fxv, Fuv

def IDFT2D(baseY, baseX, Fuv):
    Y = Fuv.shape[1]
    X = Fuv.shape[2]
    U = X
    V = Y

    # F(u,y) = (1/V) * Σv:[0~V-1] F(u,v) * eps(i * (pi * 2 * v * y))
    # F(u,v) * eps(i * (pi_2 * v * y))
    # = (a + b * i)(cos(wvy) + sin(wvy) * i)
    # = (a * cos(wvy) - b * sin(wvy) + (a * sin(wvy) + b * cos(wvy)) * i)
    Fuy = np.zeros((2, Y, U))
    for y in range(0, Y):
        Fuy[REAL][y] = baseY[REAL][y] @ Fuv[REAL] + baseY[IMAG][y] @ Fuv[IMAG]
        Fuy[IMAG][y] = -baseY[IMAG][y] @ Fuv[REAL] + baseY[REAL][y] @ Fuv[IMAG]
    
    Fuy *= 1.0 / V

    Fxy = np.zeros((Y, X))
    for x in range(0, X):
        # 虚部抵消，只计算实部
        Fxy[:, x] = (Fuy[REAL] @ baseX[REAL][x].reshape(X, 1)).ravel() + \
            (Fuy[IMAG] @ baseX[IMAG][x].reshape(X, 1)).ravel()
    
    Fxy *= 1.0 / U
    return Fxy, Fuy

def LowPassFilter(R:float, F:np.ndarray):
    for x in range(0, X):
        for y in range(0, Y):
            dTopLeft = (float(x) / X) ** 2 + (float(y) / Y) ** 2
            dBotomLeft = (float(x) / X) ** 2 + (float(y - Y) / Y) ** 2
            dTopRight = (float(x - X) / X) ** 2 + (float(y) / Y) ** 2
            dBotomRight = (float(x - X) / X) ** 2 + (float(y - Y) / Y) ** 2
            if dTopLeft > R ** 2 and dBotomLeft > R ** 2 and \
                dTopRight > R ** 2 and dBotomRight > R ** 2 :
                F[REAL][y][x] = 0.0
                F[IMAG][y][x] = 0.0

def HighPassFilter(R:float, F:np.ndarray):
    for x in range(0, X):
        for y in range(0, Y):
            dTopLeft = (float(x) / X) ** 2 + (float(y) / Y) ** 2
            dBotomLeft = (float(x) / X) ** 2 + (float(y - Y) / Y) ** 2
            dTopRight = (float(x - X) / X) ** 2 + (float(y) / Y) ** 2
            dBotomRight = (float(x - X) / X) ** 2 + (float(y - Y) / Y) ** 2
            if dTopLeft < R ** 2 or dBotomLeft < R ** 2 or \
                dTopRight < R ** 2 or dBotomRight < R ** 2 :
                F[REAL][y][x] = 0.0
                F[IMAG][y][x] = 0.0

if __name__ == '__main__':
    np.set_printoptions(precision=3)

    # 读图、初始化数据
    originImage = ReadImage(FilePath)
    Y = originImage.shape[0]
    X = originImage.shape[1]
    U = X
    V = Y
    Y_2 = int(Y/2)
    X_2 = int(X/2)

    # 基图像
    tx = np.linspace(0, X - 1, X)
    ty = np.linspace(0, Y - 1, Y)
    baseY = BaseImage(ty)
    baseX = BaseImage(tx)

    # 2D DFT
    Fxv, Fuv = DFT2D(baseY, baseX, originImage)
    FuvMin, FuvMax = np.percentile(AbsComplex(Fuv), [5, 95])

    if FilterRad >= 0:
        LowPassFilter(FilterRad, Fuv)
        # HighPassFilter(0.02, Fuv)

    # 2D IDFT
    Fxy, Fuy = IDFT2D(baseY, baseX, Fuv)

    F = np.fft.fft2(originImage)
    print(F.shape)

    # 显示图像
    fig = plt.figure(figsize=(14, 6))
    fig.subplots_adjust(
        left=0.03,    # 左边缘留白（默认~0.125）
        right=0.97,   # 右边缘留白（默认~0.9）
        bottom=0.05,  # 下边缘留白（默认~0.1）
        top=0.95,     # 上边缘留白（默认~0.9）
        wspace=0.2,  # 子图水平间距（默认~0.2）
        hspace=0.2   # 子图垂直间距（默认~0.2）
    )

    # 原信号
    ax1=fig.add_subplot(2, 4, 1)
    ax1.set_title(f"origin", loc="center", fontsize=10)
    ax1.imshow(originImage, cmap='gray')

    # 基图像
    ax2=fig.add_subplot(2, 4, 2)
    ax2.set_title(f"base Y", loc="center", fontsize=10)
    dispImg = np.zeros((baseY.shape[1], baseY.shape[2], 3))
    dispImg[:, :, 0] = baseY[0, :, :] * 0.5 + 0.5
    dispImg[:, :, 2] = baseY[1, :, :] * 0.5 + 0.5
    ax2.imshow(dispImg)

    # 单个基函数
    displayV = 10
    ax3=fig.add_subplot(2, 4, 3)
    ax3.set_title(f"base v={displayV}", loc="center", fontsize=10)
    ax3.plot(ty, baseY[REAL][displayV], color='red', linestyle='-')
    ax3.plot(ty, baseY[IMAG][displayV], color='blue', linestyle='-')

    # Fxv
    ax4=fig.add_subplot(2, 4, 4)
    ax4.set_title(f"F(x,v)", loc="center", fontsize=10)
    dispImg = AbsComplex(Fxv)
    dispImg = np.vstack([dispImg[Y_2:Y-1, :], dispImg[0:Y_2-1, :]])
    vmin, vmax = np.percentile(dispImg, [5, 95])
    ax4.imshow(dispImg, cmap='gray', vmin=vmin, vmax=vmax)

    # 基图像
    ax6=fig.add_subplot(2, 4, 6)
    dispImg = np.zeros((baseX.shape[1], baseX.shape[2], 3))
    dispImg[:, :, 0] = baseX[0, :, :] * 0.5 + 0.5
    dispImg[:, :, 2] = baseX[1, :, :] * 0.5 + 0.5
    ax6.imshow(dispImg)

    # 单个基函数
    displayU = 5
    ax7=fig.add_subplot(2, 4, 7)
    ax7.set_title(f"base u={displayU}", loc="center", fontsize=10)
    ax7.plot(tx, baseX[REAL][displayU], color='red', linestyle='-')
    ax7.plot(tx, baseX[IMAG][displayU], color='blue', linestyle='-')

    # Fuv
    ax8=fig.add_subplot(2, 4, 8)
    ax8.set_title(f"DFT F(u,v)", loc="center", fontsize=10)
    dispImg = AbsComplex(Fuv)
    dispImg = np.vstack([dispImg[Y_2:Y-1, :], dispImg[0:Y_2-1, :]])
    dispImg = np.hstack([dispImg[:, X_2:X-1], dispImg[:, 0:X_2-1]])
    ax8.imshow(dispImg, cmap='gray', vmin=FuvMin, vmax=FuvMax)

    # Fuv use numpy
    # ax5=fig.add_subplot(2, 4, 5)
    # dispImg = AbsComplex(Fuy)
    # dispImg = np.hstack([dispImg[:, X_2:X-1], dispImg[:, 0:X_2-1]])
    # vmin, vmax = np.percentile(dispImg, [5, 95])
    # ax5.imshow(dispImg, cmap='gray', vmin=vmin, vmax=vmax)

    # idft
    ax5=fig.add_subplot(2, 4, 5)
    ax5.set_title(f"IDFT", loc="center", fontsize=10)
    ax5.imshow(Fxy, cmap='gray')

    plt.show()