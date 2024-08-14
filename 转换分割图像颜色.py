import os

from PIL import Image
import numpy as np

def change_colors(image_path, output_path):
    # 打开图片
    with Image.open(image_path) as img:
        # 将图片转换为可编辑的模式
        img = img.convert("RGB")
        pixels = img.load()

        # 获取图片尺寸
        width, height = img.size

        # 遍历每个像素
        for x in range(width):
            for y in range(height):
                r, g, b = pixels[x, y]

                # 检查并更改颜色
                if r == 255 and g == 255 and b == 255:
                    # 白色变绿色
                    pixels[x, y] = (0, 255, 0)
                elif r == 0 and g == 255 and b == 0:
                    # 绿色变黄色
                    pixels[x, y] = (255, 255, 0)

        # 保存更改后的图片
        img.save(output_path)

# 使用函数
path = r'D:\2021\wwww\experiment\abalation\IMAGE_result\DRIVE\baseline+MAFF'
list = os.listdir(path)
img_list = [os.path.join(path,i) for i in list if i[0]=='c']
for i in img_list:
    change_colors(i, i)
