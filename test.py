import os
import cv2
import time
import torch
import numpy as np
from PIL import Image
from glob import glob
import torchvision.transforms as transforms
from models.UNets.DFMNet_UNet_skl4 import Model as Net

def run_test(dataset_name, data_root, test_out_path):
    test_folder = data_root
    path_ir = os.path.join(test_folder, dataset_name.split('-')[0])
    path_vi = os.path.join(test_folder, dataset_name.split('-')[1])
    path_save = os.path.join(test_out_path, dataset_name)
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    data_len = len(os.listdir(os.path.join(test_folder, dataset_name.split('-')[0])))
    print(f'ir path: {path_ir}, data len: {data_len}')
    print(f'vi path: {path_vi}, data len: {data_len}')
    print(f'save path: {path_save}, data len: {data_len}')
    path_model = '/models_pth/best_VIF.pth'  # best_MIF.pth
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Net().to(device)
    model.load_state_dict(torch.load(path_model))
    Time = []

    fused_names = [f for f in os.listdir(path_save) if os.path.isfile(os.path.join(path_save, f))]

   
    for imgname in os.listdir(path_ir):
        if imgname in fused_names:
            print(f"{imgname} tested, next image!")
            continue
        print(f"{imgname} testing!")
        filepath_ir = os.path.join(path_ir, imgname)
        filepath_vi = os.path.join(path_vi, imgname)
        # visible_image = Image.open(filepath_vi).convert ('L')  # PRT/CT/SPECT
        # infrared_image = Image.open(filepath_ir).convert ('L') # MRI
        visible_image = cv2.imread(filepath_vi, cv2.IMREAD_GRAYSCALE)
        infrared_image = cv2.imread(filepath_ir, cv2.IMREAD_GRAYSCALE)
        tran = transforms.ToTensor()
        visible_image = tran(visible_image).unsqueeze(0).cuda()
        infrared_image = tran(infrared_image).unsqueeze(0).cuda()
        img_cat = torch.cat((visible_image, infrared_image), 1)
        model.eval()
        tic = time.time()
        with torch.no_grad():
            output = model(visible_image, infrared_image, img_cat)
        end = time.time()
        Time.append(end - tic)
        output = output.cpu().numpy().squeeze() * 255  # MDRANet
        output = np.clip(output, 0, 255)
        cv2.imwrite('{}/{}'.format(path_save, imgname), output)
    Time = Time[2:len(Time) - 2]
    print(f"test time: {sum(Time)/data_len}")

if __name__ == "__main__":
    dataset = 'M3FD'  # TNO M3FD RoadScene
    data_root = f'/media/sata1/hedan/{dataset}/'
    test_out_folder = f'M3FD/'
    dataset_name = 'ir-vi'
    print(f"{dataset} testing!!!!!")
    test_time_avg = run_test(dataset_name, data_root, test_out_folder+f'{dataset}')