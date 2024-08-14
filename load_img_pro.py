def load_img_pro(img_path):
    img=Image.open(img_path)
    #img=img.resize((512,512))
    img_npy=np.array(img).transpose(2,0,1).astype(np.float32)
    for i in range(img_npy.shape[0]):
        img_npy[i]=(img_npy[i]-img_npy.mean())/img_npy[i].std()

    return img_npy