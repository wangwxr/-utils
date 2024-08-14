# 预测调色盘
# 前面是ground truth，后面是prediction
# 0,0->0设为黑
# 意思是ground truth该像素点为0，预测结果也是0
# 1,0->1设为黄
# 0,1->2设为红
# 1,1->3设为绿
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

color_img = generate_color_img(
    ground_truth=transforms.ToTensor()(ground_truth.convert('1')).to(torch.int64),
    prediction=argmax_output)

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
    output_img = Image.fromarray(palette[output_tensor.squeeze().numpy()])
    # 存储
    if file_name != '':
        output_img.save(file_name)
    return output_img