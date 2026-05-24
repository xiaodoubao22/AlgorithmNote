import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import namedtuple

vec2 = namedtuple('vec2', ['x', 'y'])

FilePathA = "../data/game_sequence/0010.png"
FilePathB = "../data/game_sequence/0011.png"
BLOCK_SIZE = 32
CORE_SIZE = 8
SIDE_SIZE = (BLOCK_SIZE - CORE_SIZE) // 2
DOWN_SAMPLE_COUNT = 2

def ReadImage(filePath:str) -> np.ndarray:
    print(type(filePath))
    readImage = Image.open(filePath)
    print(f"格式: {readImage.format}, 大小: {readImage.size}, 模式: {readImage.mode}")
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

    print(paddingImage.shape)
    
    blockCountH = ((h - 1) // CORE_SIZE + 1)
    blockCountW = ((w - 1) // CORE_SIZE + 1)
    print(blockCountH, blockCountW)

    blockStepH = h // blockCountH
    blockStepW = w // blockCountW
    print(blockStepH, blockStepW)

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


if __name__ == '__main__':
    np.set_printoptions(precision=3)

    
    window = np.hanning(BLOCK_SIZE)[:, np.newaxis] * np.hanning(BLOCK_SIZE)


    # 读图、初始化数据
    metaDataA = ReadImage(FilePathA)
    metaDataA = np.dot(metaDataA[..., :3], [0.299, 0.587, 0.114]).astype(metaDataA.dtype)
    for _ in range(DOWN_SAMPLE_COUNT):
        metaDataA = DownSample(metaDataA)
    
    h, w = metaDataA.shape
    print(f"下采样 {metaDataA.shape}")

    # 分块
    blocksImage, bch, bcw = BlockGenerate(metaDataA)

    # window
    for bh in range(bch):
        for bw in range(bcw):
            blocksImage[bh * BLOCK_SIZE:bh * BLOCK_SIZE + BLOCK_SIZE, bw * BLOCK_SIZE:bw * BLOCK_SIZE + BLOCK_SIZE] *= window


    # FFT
    FFTImage = np.zeros([bch * BLOCK_SIZE, bcw * BLOCK_SIZE], dtype=np.complex128)
    for bh in range(bch):
        for bw in range(bcw):
            FFTImage[bh * BLOCK_SIZE:bh * BLOCK_SIZE + BLOCK_SIZE, bw * BLOCK_SIZE:bw * BLOCK_SIZE + BLOCK_SIZE] =\
            np.fft.fft2(blocksImage[bh * BLOCK_SIZE:bh * BLOCK_SIZE + BLOCK_SIZE, bw * BLOCK_SIZE:bw * BLOCK_SIZE + BLOCK_SIZE])





    # 读图、初始化数据
    metaDataB = ReadImage(FilePathB)
    metaDataB = np.dot(metaDataB[..., :3], [0.299, 0.587, 0.114]).astype(metaDataB.dtype)
    for _ in range(DOWN_SAMPLE_COUNT):
        metaDataB = DownSample(metaDataB)
    
    h, w = metaDataB.shape
    print(f"下采样 {metaDataB.shape}")

    # 分块
    blocksImageB, bch, bcw = BlockGenerate(metaDataB)

    # window
    for bh in range(bch):
        for bw in range(bcw):
            blocksImageB[bh * BLOCK_SIZE:bh * BLOCK_SIZE + BLOCK_SIZE, bw * BLOCK_SIZE:bw * BLOCK_SIZE + BLOCK_SIZE] *= window

    # FFT
    FFTImageB = np.zeros([bch * BLOCK_SIZE, bcw * BLOCK_SIZE], dtype=np.complex128)
    for bh in range(bch):
        for bw in range(bcw):
            FFTImageB[bh * BLOCK_SIZE:bh * BLOCK_SIZE + BLOCK_SIZE, bw * BLOCK_SIZE:bw * BLOCK_SIZE + BLOCK_SIZE] =\
            np.fft.fft2(blocksImageB[bh * BLOCK_SIZE:bh * BLOCK_SIZE + BLOCK_SIZE, bw * BLOCK_SIZE:bw * BLOCK_SIZE + BLOCK_SIZE])

    # CPS
    CPSImage = FFTImage * np.conj(FFTImageB)
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
            print(maxVal, x, y)
            OFImage[bh, bw] = np.array([x, y]) / 32.0


    dx = OFImage[:,:,0]
    dy = OFImage[:,:,1]

    # 计算角度和幅度
    angle = np.arctan2(dy, dx)
    magnitude = np.sqrt(dx**2 + dy**2)

    # 归一化
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)

    # 构建 HSV
    hsv = np.zeros((bch,bcw,3))
    hsv[...,0] = (angle + np.pi) / (2 * np.pi)  # 色相
    hsv[...,1] = magnitude                     # 饱和度
    hsv[...,2] = 1.0                            # 亮度

    # 转 RGB
    from matplotlib.colors import hsv_to_rgb
    flow_rgb = hsv_to_rgb(hsv)

    plt.imshow(flow_rgb)
    plt.title('Optical Flow (Matplotlib)')
    plt.axis('off')
    plt.show()

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
    ax.set_title(f"blockA {blocksImage.shape}", loc="center", fontsize=10)
    ax.imshow(blocksImage, cmap='gray')

    ax=fig.add_subplot(2, 4, 5)
    ax.set_title(f"blockB", loc="center", fontsize=10)
    ax.imshow(blocksImageB, cmap='gray')

    # FFT
    ax=fig.add_subplot(2, 4, 2)
    ax.set_title(f"FFT A", loc="center", fontsize=10)
    fAbs = np.abs(FFTImage)
    fMin, fMax = np.percentile(fAbs, [5, 95])
    ax.imshow(fAbs, cmap='gray', vmin=fMin, vmax=fMax)

    ax=fig.add_subplot(2, 4, 6)
    ax.set_title(f"FFT B", loc="center", fontsize=10)
    fAbs = np.abs(FFTImageB)
    fMin, fMax = np.percentile(fAbs, [5, 95])
    ax.imshow(fAbs, cmap='gray', vmin=fMin, vmax=fMax)

    # CPS
    ax=fig.add_subplot(2, 4, 3)
    ax.set_title(f"CPS", loc="center", fontsize=10)
    fAbs = np.abs(CPSImage)
    fMin, fMax = np.percentile(fAbs, [5, 95])
    ax.imshow(fAbs, cmap='gray', vmin=fMin, vmax=fMax)

    # IFFT
    ax=fig.add_subplot(2, 4, 7)
    ax.set_title(f"CPS IFFT", loc="center", fontsize=10)
    ax.imshow(cpsIFFT.real)

    # OFX
    ax=fig.add_subplot(2, 4, 4)
    ax.set_title(f"OFX", loc="center", fontsize=10)
    ax.imshow(OFImage[:,:,0])

    # OFY
    ax=fig.add_subplot(2, 4, 8)
    ax.set_title(f"OFY", loc="center", fontsize=10)
    ax.imshow(OFImage[:,:,1])

    plt.show()



# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import hsv_to_rgb

# # ======================
# # 1. 构造测试数据 (N帧 32x32x2)
# # ======================
# N = 10  # 帧数
# frames = np.random.randn(N, 32, 32, 2)  # 你的数据格式
# current_idx = 0  # 当前帧

# # ======================
# # 2. 光流转彩色图函数
# # ======================
# def flow2rgb(flow):
#     dx = flow[..., 0]
#     dy = flow[..., 1]
#     angle = np.arctan2(dy, dx)
#     mag = np.sqrt(dx**2 + dy**2)
#     mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
    
#     hsv = np.zeros((32, 32, 3))
#     hsv[..., 0] = (angle + np.pi) / (2 * np.pi)
#     hsv[..., 1] = mag
#     hsv[..., 2] = 1.0
#     return hsv_to_rgb(hsv)

# # ======================
# # 3. 创建 SUBPLOT 多子图
# # 示例：2行1列 → 同时显示 光流彩色图 + 通道1热力图
# # ======================
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))  # 1行2列

# # 初始化显示第0帧
# flow_rgb = flow2rgb(frames[current_idx])
# ch1 = frames[current_idx, ..., 0]

# # 子图1：光流彩色图
# im1 = ax1.imshow(flow_rgb)
# ax1.set_title(f"Flow Frame {current_idx+1}/{N}")
# ax1.axis('off')

# # 子图2：通道1热力图
# im2 = ax2.imshow(ch1, cmap='jet')
# ax2.set_title(f"Channel 1")
# ax2.axis('off')

# # ======================
# # 4. 键盘按键刷新所有子图
# # ======================
# def on_key(event):
#     global current_idx
    
#     # 右键 → 下一帧
#     if event.key == 'right':
#         current_idx = min(current_idx + 1, N - 1)
#     # 左键 → 上一帧
#     elif event.key == 'left':
#         current_idx = max(current_idx - 1, 0)
    
#     # 获取当前帧数据
#     frame = frames[current_idx]
#     new_rgb = flow2rgb(frame)
#     new_ch1 = frame[..., 0]
    
#     # 更新所有子图
#     im1.set_data(new_rgb)
#     im2.set_data(new_ch1)
    
#     # 更新标题
#     ax1.set_title(f"Flow Frame {current_idx+1}/{N}")
    
#     # 刷新画布
#     fig.canvas.draw()

# # 绑定键盘事件
# fig.canvas.mpl_connect('key_press_event', on_key)

# plt.tight_layout()
# plt.show()