import numpy as np
from skimage import io, morphology
from skimage.measure import label
import cv2

def extract_crosspoints(skel, cross_kernel):
    filtered = cv2.filter2D(skel.astype(np.uint8), -1, cross_kernel)
    crosspoints = filtered >= 13
    return crosspoints


def extract_endpoints(skel, cross_kernel):
    filtered = cv2.filter2D(skel.astype(np.uint8), -1, cross_kernel)
    endpoints = filtered == 11
    return endpoints


def apply_colors_to_skeleton(skeleton, sparsified_points, endpoints, crosspoints):
    red_channel = skeleton.copy()
    green_channel = skeleton.copy()
    blue_channel = skeleton.copy()

    for y, x in np.argwhere(sparsified_points == 1):
        red_channel[y, x] = 255
        green_channel[y, x] = 0
        blue_channel[y, x] = 0

    for y, x in np.argwhere(endpoints):
        green_channel[y, x] = 255

    for y, x in np.argwhere(crosspoints):
        blue_channel[y, x] = 255

    colored_skeleton = np.stack([red_channel, green_channel, blue_channel], axis=-1)
    return colored_skeleton
def sparsify_skeleton(skeleton, interval=10):
    labeled_skeleton = label(skeleton)
    sparsified_skeleton = np.zeros_like(skeleton)
    for region_label in np.unique(labeled_skeleton):
        if region_label == 0:
            continue
        coords = np.argwhere(labeled_skeleton == region_label)
        selected_indices = np.arange(0, len(coords), interval)
        selected_coords = coords[selected_indices]
        for y, x in selected_coords:
            sparsified_skeleton[y, x] = 1
    return sparsified_skeleton


def apply_red_to_skeleton(skeleton, sparsified_points):
    red_channel = skeleton.copy()
    green_channel = skeleton.copy()
    blue_channel = skeleton.copy()

    for y, x in np.argwhere(sparsified_points == 1):
        red_channel[y, x] = 255  # Red
        green_channel[y, x] = 0  # Green
        blue_channel[y, x] = 0  # Blue

    colored_skeleton = np.stack([red_channel, green_channel, blue_channel], axis=-1)
    return colored_skeleton


# Load and process the skeleton
skeleton_image_path = r'D:\some CV\try by myself\后处理\cleaned_skeleton.png'  # Change this to your skeleton image path
skeleton_image = io.imread(skeleton_image_path, as_gray=True)
skeleton_image = skeleton_image > 0  # Convert to binary image if not already
cross_kernel = np.array([[1, 1, 1],
                             [1, 10, 1],
                             [1, 1, 1]], dtype=np.uint8)
# Sparsify the skeleton
#todo:通过interval来控制点的稠密 初始为10
sparsified_points = sparsify_skeleton(skeleton_image, interval=6)
endpoints = extract_endpoints(skeleton_image,cross_kernel=cross_kernel)
crosspoints = extract_crosspoints(skeleton_image,cross_kernel=cross_kernel)
# Apply red color to the sparsified points on the original skeleton
sparsified_points = crosspoints+endpoints+sparsified_points
colored_skeleton = apply_red_to_skeleton(skeleton_image, sparsified_points)

# Save the result
save_path = 'xishu6.png'  # Change this to your desired save path
io.imsave(save_path, colored_skeleton)

print(f"Colored skeleton saved to {save_path}")
#提取这些点的过程主要分为两个步骤：稀疏化处理和点的提取。
#
# 1. **稀疏化处理**：我使用了一个自定义的函数`sparsify_skeleton`来均匀地从每个骨架分支上选择点。这个过程涉及以下关键操作：
#
#    - **标记连通区域**：使用`label`函数标记骨架图中的连通区域（分支）。每个连通区域视为一个独立的分支。
#    - **均匀选择点**：对于每个连通区域，我获取其所有像素点的坐标，然后基于指定的间隔（`interval`参数）从这些坐标中均匀地选择点。间隔值越大，提取的点越稀疏。
#
# 2. **点的提取**：通过上述稀疏化处理，我们得到了一个包含所选点位置的二维数组（`sparsified_skeleton`），其中值为1的位置表示被选中的点。
#
# 接下来，为了将这些点在原始骨架图上标为红色，我进行了以下操作：
#
# - 创建了原始骨架图的三个副本，分别代表红、绿、蓝三个颜色通道。由于原始骨架是二维的，所以初始时这三个通道的内容是相同的。
# - 遍历`sparsified_skeleton`中所有值为1的位置，即我们之前选出的点。对于这些点的位置，我在红色通道中将对应的值设置为255（表示红色），并在绿色和蓝色通道中将对应的值设置为0，从而将这些点标记为红色。
# - 最后，将这三个通道合并成一个三维数组（彩色图像），其中包含了原始骨架以及用红色标记的稀疏化点。
#
# 这个处理流程的结果是一个在保留了原始骨架结构的同时，将关键点以红色高亮显示的彩色图像，适用于进一步分析或可视化展示。