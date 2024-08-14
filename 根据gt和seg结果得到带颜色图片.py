import os.path
import torch
import numpy as np
from PIL import Image
VESSEL_PALETTE = np.asarray(
    [
        # 黑
        [0, 0, 0],
        # 黄(少分)黄
        [255, 255, 0],
        # 红（多分）
        [255, 0, 0],
        # 绿
        [0, 255, 0],
    ], dtype=np.uint8
)



def generate_color_img(ground_truth, prediction, file_name='', palette=VESSEL_PALETTE):
    '''
    :param ground_truth:tensor[1,W,H]，输入的groundtruth，每个像素的值为0或1且为整数，
                    建议使用transforms.ToTensor()(Image.open(picture_name).convert('1')).to(torch.int64)读入
    :param prediction:tensor[1,W,H]，网络的输出，每个像素的值为0或1且为整数
    :param file_name:str，要存储的话，存储的文件路径
    :param palette:numpy，调色板
    :return:Iamge，经过调色板调色的图片
    '''
    output_tensor = ground_truth + 2 * prediction
    # output_img = Image.fromarray(palette[output_tensor.squeeze().numpy().astype(np.uint(8))])
    output_img = Image.fromarray(palette[output_tensor.numpy().astype(np.uint(8))])
    # 存储
    if file_name != '':
        output_img.save(file_name)
    return output_img
path = r'D:\2021\wwww\experiment\对比实验\MCDAU-net\CHASEDB1'
labelpath = os.path.join(path,'label')
resultpath = os.path.join(path,'result')
coloredpath = os.path.join(path,'colored')

labelfile_list = os.listdir(labelpath)
resultfile_list = os.listdir(resultpath)
label_list = [os.path.join(labelpath,i) for i in labelfile_list]
result_list = [os.path.join(resultpath,i) for i in resultfile_list]
file=''
for i in range(len(label_list)):
    gt = torch.tensor(np.array(Image.open(label_list[i]).convert('L').resize((512,512),Image.NEAREST)))/255
    seg = torch.tensor(np.array(Image.open(result_list[i]).convert('L').resize((512,512),Image.NEAREST)))/255
    fname = os.path.join(coloredpath,resultfile_list[i])
    generate_color_img(gt,seg,fname)
#
#
# color_img = generate_color_img(
#     ground_truth=transforms.ToTensor()(ground_truth.convert('1')).to(torch.int64),
#     prediction=argmax_output)