import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import namedtuple
from matplotlib.colors import hsv_to_rgb

vec2 = namedtuple('vec2', ['x', 'y'])

FilePathA = "../data/game_sequence/0010.png"
FilePathB = "../data/game_sequence/0011.png"
FILE_PREFIX = "../data/game_sequence/"
BLOCK_SIZE = 32
CORE_SIZE = 8
SIDE_SIZE = (BLOCK_SIZE - CORE_SIZE) // 2
DOWN_SAMPLE_COUNT = 2

metaData = []
window = np.hanning(BLOCK_SIZE)[:, np.newaxis] * np.hanning(BLOCK_SIZE)
preColorImage = np.zeros([0])
curColorImage = np.zeros([0])
preDownSampleImage = np.zeros([0])
curDownSampleImage = np.zeros([0])
preBlocksImage = np.zeros([0])
curBlocksImage = np.zeros([0])
preFFTImage = np.zeros([0])
curFFTImage = np.zeros([0])
CPSImage = np.zeros([0])
cpsIFFT = np.zeros([0])
OFImage = np.zeros([0])

def ReadImage(filePath:str) -> np.ndarray:
    readImage = Image.open(filePath)
    print(f"格式: {readImage.format}, 大小: {readImage.size}, 模式: {readImage.mode} 路径：{filePath}")
    return np.array(readImage) / 255.0

def DownSample(oriImage:np.ndarray) -> np.ndarray:
    # 确保图像的高和宽是 2 的倍数（如果不是，建议先切片成偶数）
    h, w = oriImage.shape
    h_even, w_even = h - (h % 2), w - (w % 2)
    clipped = oriImage[:h_even, :w_even]

    # 重塑维度为 (h/2, 2, w/2, 2)，然后在第 1 和第 3 个轴上取均值
    downsampled = clipped.reshape(h_even // 2, 2, w_even // 2, 2).mean(axis=(1, 3))
    
    # 根据需要转换回原图像的数据类型（例如 uint8）
    return downsampled.astype(oriImage.dtype)

def BlockGenerate(oriImage:np.ndarray) -> tuple[np.ndarray, int, int]:
    h, w = oriImage.shape

    paddingImage = np.vstack((oriImage[:SIDE_SIZE, :], oriImage, oriImage[-SIDE_SIZE:, :]))
    paddingImage = np.hstack((paddingImage[:, :SIDE_SIZE], paddingImage, paddingImage[:, -SIDE_SIZE:]))
    
    blockCountH = ((h - 1) // CORE_SIZE + 1)
    blockCountW = ((w - 1) // CORE_SIZE + 1)
    print("分块数量", blockCountH, blockCountW)

    blockStepH = h // blockCountH
    blockStepW = w // blockCountW
    print("块间距", blockStepH, blockStepW)

    blocksH = blockCountH * BLOCK_SIZE
    blocksW = blockCountW * BLOCK_SIZE
    blocks = np.zeros([blocksH, blocksW])
    stH = 0
    stW = 0
    for bh in range(blockCountH):
        stW = 0
        for bw in range(blockCountW):
            blocks[bh * BLOCK_SIZE:bh * BLOCK_SIZE + BLOCK_SIZE, bw * BLOCK_SIZE:bw * BLOCK_SIZE + BLOCK_SIZE] =\
            paddingImage[stH:stH + BLOCK_SIZE, stW:stW + BLOCK_SIZE]
            stW += blockStepW
        stH += blockStepH
    return blocks, blockCountH, blockCountW

def FlowToHsv(flow:np.ndarray) -> np.ndarray:
    dx = flow[..., 0]
    dy = flow[..., 1]

    # 计算角度和幅度
    angle = np.arctan2(dy, dx)
    magnitude = np.sqrt(dx**2 + dy**2)

    # 归一化
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)

    # 构建 HSV
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3))
    hsv[...,0] = (angle + np.pi) / (2 * np.pi)  # 色相
    hsv[...,1] = magnitude                      # 饱和度
    hsv[...,2] = 1.0                            # 亮度

    # 转 RGB
    return hsv_to_rgb(hsv)

def Loop(n:int, firstLoop:bool):
    print("======循环开始======")
    global metaData
    global window
    global preColorImage
    global curColorImage
    global preDownSampleImage
    global curDownSampleImage
    global preBlocksImage
    global curBlocksImage
    global preFFTImage
    global curFFTImage
    global CPSImage
    global cpsIFFT
    global OFImage

    if not firstLoop:
        preColorImage, curColorImage = curColorImage, preColorImage
        preDownSampleImage, curDownSampleImage = curDownSampleImage, preDownSampleImage
        preBlocksImage, curBlocksImage = curBlocksImage, preBlocksImage
        preFFTImage, curFFTImage = curFFTImage, preFFTImage
        preColorImage, curColorImage = curColorImage, preColorImage

    # 取数据 
    curColorImage = metaData[n]
    
    # 转灰度图
    lumaImage = np.dot(curColorImage[..., :3], [0.299, 0.587, 0.114]).astype(curColorImage.dtype)

    # 下采样
    curDownSampleImage = lumaImage
    for _ in range(DOWN_SAMPLE_COUNT):
        curDownSampleImage = DownSample(curDownSampleImage)
    print(f"下采样尺寸 {curDownSampleImage.shape}")

    # 分块
    curBlocksImage, bch, bcw = BlockGenerate(curDownSampleImage)

    # 加窗
    for bh in range(bch):
        for bw in range(bcw):
            curBlocksImage[bh * BLOCK_SIZE:bh * BLOCK_SIZE + BLOCK_SIZE, bw * BLOCK_SIZE:bw * BLOCK_SIZE + BLOCK_SIZE] *= window

    # FFT
    curFFTImage = np.zeros([bch * BLOCK_SIZE, bcw * BLOCK_SIZE], dtype=np.complex128)
    for bh in range(bch):
        for bw in range(bcw):
            curFFTImage[bh * BLOCK_SIZE:bh * BLOCK_SIZE + BLOCK_SIZE, bw * BLOCK_SIZE:bw * BLOCK_SIZE + BLOCK_SIZE] =\
            np.fft.fft2(curBlocksImage[bh * BLOCK_SIZE:bh * BLOCK_SIZE + BLOCK_SIZE, bw * BLOCK_SIZE:bw * BLOCK_SIZE + BLOCK_SIZE])

    if firstLoop:
        print("======首帧循环结束======")
        return

    # CPS
    CPSImage = curFFTImage * np.conj(preFFTImage)
    CPSImage /= np.abs(CPSImage) + 1e-6

    # IFFT
    cpsIFFT = np.zeros([bch * BLOCK_SIZE, bcw * BLOCK_SIZE], dtype=np.complex128)
    for bh in range(bch):
        for bw in range(bcw):
            cpsIFFT[bh * BLOCK_SIZE:bh * BLOCK_SIZE + BLOCK_SIZE, bw * BLOCK_SIZE:bw * BLOCK_SIZE + BLOCK_SIZE] =\
            np.fft.ifft2(CPSImage[bh * BLOCK_SIZE:bh * BLOCK_SIZE + BLOCK_SIZE, bw * BLOCK_SIZE:bw * BLOCK_SIZE + BLOCK_SIZE])

    OFImage = np.zeros([bch, bcw, 2], dtype=float)
    # OF
    for bh in range(bch):
        for bw in range(bcw):
            colleration = cpsIFFT[bh * BLOCK_SIZE:bh * BLOCK_SIZE + BLOCK_SIZE, bw * BLOCK_SIZE:bw * BLOCK_SIZE + BLOCK_SIZE].real
            maxVal = np.max(colleration)
            y, x = np.unravel_index(np.argmax(colleration), colleration.shape)
            if y >= BLOCK_SIZE / 2:
                y -= BLOCK_SIZE
            if x >= BLOCK_SIZE / 2:
                x -= BLOCK_SIZE
            OFImage[bh, bw] = np.array([x, y]) / 32.0
    print("======循环结束======")

if __name__ == '__main__':
    np.set_printoptions(precision=3)

    start = 10
    end = 19
    metaData = [ReadImage(FILE_PREFIX + "{:04d}.png".format(n)) for n in range(start, end + 1)]
    print(f"连续图像数量 {len(metaData)}")
    print(metaData[0].shape)

    curIndex = 0

    Loop(curIndex, True)
    curIndex += 1

    Loop(curIndex, False)
    curIndex += 1

    # 显示图像
    fig, ax = plt.subplots(2, 3, figsize=(18, 9))  # 1行2列
    fig.subplots_adjust(
        left=0.03,    # 左边缘留白（默认~0.125）
        right=0.97,   # 右边缘留白（默认~0.9）
        bottom=0.05,  # 下边缘留白（默认~0.1）
        top=0.95,     # 上边缘留白（默认~0.9）
        wspace=0.2,  # 子图水平间距（默认~0.2）
        hspace=0.2   # 子图垂直间距（默认~0.2）
    )
    im = [() for _ in range(2)]

    # color：
    pRow, pCol = 0, 0
    im[pRow] += (ax[pRow][pCol].imshow(curColorImage),)
    ax[pRow][pCol].set_title(f"blocksImage {curIndex}")
    ax[pRow][pCol].axis('off')

    # downSample
    pRow, pCol = 0, 1
    im[pRow] += (ax[pRow][pCol].imshow(curDownSampleImage, cmap='gray'),)
    ax[pRow][pCol].set_title(f"down sample {curDownSampleImage.shape}")
    ax[pRow][pCol].axis('off')

    # block：
    pRow, pCol = 0, 2
    im[pRow] += (ax[pRow][pCol].imshow(curBlocksImage, cmap='gray'),)
    ax[pRow][pCol].set_title(f"block {curBlocksImage.shape}")
    ax[pRow][pCol].axis('off')

    # FFT
    pRow, pCol = 1, 0
    fAbs = np.abs(curFFTImage)
    fMin, fMax = np.percentile(fAbs, [5, 95])
    im[pRow] += (ax[pRow][pCol].imshow(fAbs, cmap='gray', vmin=fMin, vmax=fMax),)
    ax[pRow][pCol].set_title(f"FFT {fAbs.shape}")
    ax[pRow][pCol].axis('off')

    # IFFT
    pRow, pCol = 1, 1
    im[pRow] += (ax[pRow][pCol].imshow(cpsIFFT.real,),)
    ax[pRow][pCol].set_title(f"CPS IFFT")
    ax[pRow][pCol].axis('off')

    # OFX
    pRow, pCol = 1, 2
    im[pRow] += (ax[pRow][pCol].imshow(FlowToHsv(OFImage)),)
    ax[pRow][pCol].set_title(f"OFX", loc="center", fontsize=10)
    ax[pRow][pCol].axis('off')


    def on_key(event):
        global curIndex
        
        # 右键 → 下一帧
        if event.key == 'right':
            curIndex = min(curIndex + 1, len(metaData) - 1)
        # 左键 → 上一帧
        elif event.key == 'left':
            curIndex = max(curIndex - 1, 0)
        
        # 刷新当前帧数据
        Loop(curIndex, False)
        
        # 更新所有子图
        im[0][0].set_data(curColorImage)
        im[0][1].set_data(curDownSampleImage)
        im[0][2].set_data(curBlocksImage)

        fAbs = np.abs(curFFTImage)
        fMin, fMax = np.percentile(fAbs, [5, 95])
        im[1][0].set_data(fAbs)
        im[1][0].set_clim(vmin=fMin, vmax=fMax)

        im[1][1].set_data(cpsIFFT.real)

        im[1][2].set_data(FlowToHsv(OFImage))
        
        # 刷新画布
        fig.canvas.draw()

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.tight_layout()
    plt.show()
