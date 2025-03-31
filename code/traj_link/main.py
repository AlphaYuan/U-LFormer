

import numpy as np
#from andi_datasets.utils_videos import import_tiff_video
import matplotlib.pyplot as plt
np.random.seed(0)
import csv
from PIL import Image
import numpy as np


def import_tiff_video(tiff_file_path):
    """
    Import a TIFF video file as a NumPy array.

    This function reads a multi-frame TIFF file and converts each frame into a NumPy array.
    All frames are stacked along a new axis, resulting in a 3D array if the frames are 2D
    (or a 4D array if the frames are 3D).

    Parameters
    ----------
    tiff_file_path : str
        The file path of the TIFF video file.

    Returns
    -------
    numpy.ndarray
        A NumPy array containing the video frames stacked along a new axis.
        The shape of the array is (N, M, O, ...) where N is the number of frames,
        and M, O, ... are the dimensions of each frame.

    Raises
    ------
    FileNotFoundError
        If the TIFF file specified by `tiff_file_path` does not exist.
    """

    # Open the TIFF file
    with Image.open(tiff_file_path) as img:
        # Initialize a list to hold each frame as a numpy array
        frames = []

        # Loop over each frame in the TIFF
        try:
            while True:
                # Convert the current frame to a numpy array and add to the list
                frames.append(np.array(img))

                # Move to the next frame
                img.seek(img.tell() + 1)
        except EOFError:
            # End of sequence; stop iterating
            pass

        # Stack all frames along a new axis (creating a 3D array if frames are 2D)
        video_array = np.stack(frames, axis=0)

        return video_array




def generate_data(exp_path,tif_path,fov=1):
    video = import_tiff_video(exp_path+tif_path)
    mean = np.mean(video[1])
    variance = np.var(video[1])

    import copy

    #todo: kk = video[0][y][x]    #在python中和imagej中的x，y是反的

    unique_elements = np.unique(video[0])
    unique_elements=unique_elements[:-1]

    index=[[] for _ in range(len(unique_elements))]

    for n,i in enumerate(unique_elements):
        index[n].append(i)
        indices = np.argwhere(video[0] == i)
        mean_x=0
        mean_y=0
        for j in range(len(indices)):
            mean_x=mean_x+indices[j][1]       #在python中和imagej中的x，y是反的
            mean_y = mean_y + indices[j][0]
        mean_x=mean_x/(j+1)
        index[n].append(mean_x)
        mean_y=mean_y/(j+1)
        index[n].append(mean_y)

    a=abs(video[0]-255)

    aa=copy.deepcopy(a)


    '''for i in range(len(a)):
        for j in range(len(a[0])):
            if a[i][j]!=0:
                a[i][j]=1
                if j >0:
                    a[i][j-1] = 1
                if i >0:
                    a[i-1][j] = 1
                if i > 0 and j >0:
                    a[i - 1][j-1] = 1'''

    for i in range(len(a)):
        for j in range(len(a[0])):
            if a[i][j]!=0:
                a[i][j]=1
                if j >0:
                    a[i][j-1] = 1
                    #a[i][j + 1] = 1
                if i >0:
                    a[i-1][j] = 1
                    #a[i +1][j] = 1

                if i > 0 and j >0:
                    a[i - 1][j-1] = 1
                    #a[i + 1][j + 1] = 1

    for i in range(len(a)):
        for j in range(len(a[0])):
            if a[i][j]!=0:
                aa[i][j]=1
                if j+1 < len(a[0]):
                    #a[i][j-1] = 1
                    aa[i][j + 1] = 1
                if i+1 <len(a):
                    #a[i-1][j] = 1
                    aa[i +1][j] = 1

                if j+1 < len(a[0]) and i+1 <len(a):
                    #a[i - 1][j-1] = 1
                    aa[i + 1][j + 1] = 1


    a=aa+a


    for i in range(len(a)):
        for j in range(len(a[0])):
            if a[i][j]!=0:
                a[i][j]=1


    b=a*video[1]

    for i in range(len(b)):
        for j in range(len(b[0])):
            if b[i][j]==0:
                #b[i][j]=np.clip(int(np.random.normal(mean, np.sqrt(variance))),0,100)
                b[i][j]=np.random.randint(mean-10, mean+10)

    video[0]=b
    import tifffile as tiff



    tiff.imwrite(exp_path+'for_track_fov_{}.tiff'.format(fov), video)

    with open(exp_path+'for_vip_index_fov_{}.csv'.format(fov), 'w', newline='') as file:
        writer = csv.writer(file)
        #for i in range(len(index)):
        writer.writerows(index)



from scipy.ndimage import gaussian_filter
from skimage import restoration
import matplotlib.pyplot as plt



num_exp=13
num_fov=30

for i in range (num_exp):
    for j in range (num_fov):
        public_data_path = 'public_data_validation_v1/track_1/exp_{}/'.format(i)  # make sure the folder has this name or change it
        tif_data_path = 'videos_fov_{}.tiff'.format(j)
        generate_data(exp_path=public_data_path,tif_path=tif_data_path,fov=j)















# 背景减除（高斯滤波）
'''background = gaussian_filter(array, sigma=2)
array_bg_removed = array - background

# 去噪（使用维纳滤波）
array_denoised = restoration.denoise_wavelet(array_bg_removed, channel_axis=None)

# 显示原始图像、背景图像、背景减除后的图像和去噪后的图像
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].imshow(array, cmap='gray')
axes[0].set_title('Original Image')
axes[1].imshow(background, cmap='gray')
axes[1].set_title('Background')
axes[2].imshow(array_bg_removed, cmap='gray')
axes[2].set_title('Background Removed')
axes[3].imshow(array_denoised, cmap='gray')
axes[3].set_title('Denoised Image')
for ax in axes:
    ax.axis('off')
plt.show()'''
'''
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.util import random_noise
from skimage.restoration import rolling_ball

# 假设你有一个128x128的数组，值在0-255
# 这里生成一个示例数组，实际应用中你会加载你的图像数据
#array = np.random.randint(0, 256, (128, 128), dtype=np.uint8)

# 添加一些随机噪声以模拟真实情况

# 使用rolling_ball算法减除背景
radius = 4  # 设置滚动球半径
background = rolling_ball(array, radius=radius)

# 减除背景
array_bg_removed = array - background'''

# 显示原始图像、背景图像和背景减除后的图像
'''fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(array, cmap='gray')
axes[0].set_title('Original Image')
axes[1].imshow(background, cmap='gray')
axes[1].set_title('Background')
axes[2].imshow(array_bg_removed, cmap='gray')
axes[2].set_title('Background Removed')
for ax in axes:
    ax.axis('off')
plt.show()'''


