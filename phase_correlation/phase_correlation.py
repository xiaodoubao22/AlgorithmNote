import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import namedtuple

vec2 = namedtuple('vec2', ['x', 'y'])

FilePath = "../data/images/Lena.bmp"
BLOCKSIZE = vec2(128, 128)
START_A = vec2(10, 10)
START_B = vec2(50, 80)

def ReadImage(filePath:str) -> np.ndarray:
    print(type(filePath))
    readImage = Image.open(filePath)
    print(f"格式: {readImage.format}, 大小: {readImage.size}, 模式: {readImage.mode}")
    return np.array(readImage) / 255.0

if __name__ == '__main__':
    np.set_printoptions(precision=3)

    # 读图、初始化数据
    metaData = ReadImage(FilePath)

    # 裁剪
    blockA = metaData[START_A.y : START_A.y + BLOCKSIZE.y, START_A.x : START_A.x + BLOCKSIZE.x]
    blockB = metaData[START_B.y : START_B.y + BLOCKSIZE.y, START_B.x : START_B.x + BLOCKSIZE.x]

    # FFT
    fftA = np.fft.fft2(blockA)
    fftB = np.fft.fft2(blockB)

    # CPS
    cps = fftA * np.conj(fftB)
    cps /= np.abs(cps) + 1e-6

    # IFFT
    cpsIFFT = np.fft.ifft2(cps).real

    maxVal = np.max(cpsIFFT)
    row, col = np.unravel_index(np.argmax(cpsIFFT), cpsIFFT.shape)

    # conj
    ifftB = np.fft.ifft2(np.conj(fftB)).real

    print(f"峰值强度: {maxVal}")
    print(f"峰值在矩阵中的坐标 -> 行(Y): {row}, 列(X): {col}")

    # 显示图像
    fig = plt.figure(figsize=(16, 8))
    fig.subplots_adjust(
        left=0.03,    # 左边缘留白（默认~0.125）
        right=0.97,   # 右边缘留白（默认~0.9）
        bottom=0.05,  # 下边缘留白（默认~0.1）
        top=0.95,     # 上边缘留白（默认~0.9）
        wspace=0.2,  # 子图水平间距（默认~0.2）
        hspace=0.2   # 子图垂直间距（默认~0.2）
    )

    # 原图像
    ax=fig.add_subplot(2, 4, 1)
    ax.set_title(f"blockA", loc="center", fontsize=10)
    ax.imshow(blockA, cmap='gray')

    ax=fig.add_subplot(2, 4, 5)
    ax.set_title(f"blockB", loc="center", fontsize=10)
    ax.imshow(blockB, cmap='gray')

    # FFT
    ax=fig.add_subplot(2, 4, 2)
    ax.set_title(f"FFT A", loc="center", fontsize=10)
    fAbs = np.abs(fftA)
    fMin, fMax = np.percentile(fAbs, [5, 95])
    ax.imshow(fAbs, cmap='gray', vmin=fMin, vmax=fMax)

    ax=fig.add_subplot(2, 4, 6)
    ax.set_title(f"FFT B", loc="center", fontsize=10)
    fAbs = np.abs(fftB)
    fMin, fMax = np.percentile(fAbs, [5, 95])
    ax.imshow(fAbs, cmap='gray', vmin=fMin, vmax=fMax)

    # CPS
    ax=fig.add_subplot(2, 4, 3)
    ax.set_title(f"CPS", loc="center", fontsize=10)
    fAbs = np.abs(cps)
    fMin, fMax = np.percentile(fAbs, [5, 95])
    ax.imshow(fAbs, cmap='gray', vmin=fMin, vmax=fMax)

    # IFFT
    ax=fig.add_subplot(2, 4, 7)
    ax.set_title(f"CPS IFFT", loc="center", fontsize=10)
    ax.imshow(cpsIFFT)

    ax=fig.add_subplot(2, 4, 8)
    ax.set_title(f"IFFT(conj(fftB))", loc="center", fontsize=10)
    ax.imshow(ifftB, cmap='gray')

    plt.show()