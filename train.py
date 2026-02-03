import argparse
import time
import joblib
import os
import torch
import cv2
import numpy as np
from glob import glob
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets.Datasets import FusionDataset as FD

from losses.utils import AverageMeter
import logging
from utils_other import time_to_str, tensor2img, setup_logger, tensor2uint
from evaluator import ev_one


def select_loss(loss_name):
    if loss_name == 'SwinF_loss':
        from losses.SwinF_loss import fusion_loss_med as Loss
    elif loss_name == 'DMFNet_loss':
        from losses.DMFNet_loss import Fusionloss as Loss
    return Loss()


def select_model(model_name):
    if model_name == 'DMFNet_UNet_skl4':
        from models.UNets.DFMNet_UNet_skl4 import Model as Net
    elif model_name in ['DMFNet_UNet_skl4_xrsy']:
        # 'all_BN_33', 'all_BN_LKC', 'IN_GN_BN_33'
        from models.UNets.DFMNet_UNet_skl_xrsy import Model as Net
    return Net()


def run_train():
    # 模型保存路径
    test_out_path = f'./results/{opt.model_name}/{opt.loss_type}_{localtime}/' # 融合结果保存路径
    model_out_path = f"./models_pth/{opt.model_name}/{opt.loss_type}_{localtime}/"
    if not os.path.exists(model_out_path):
        os.makedirs(model_out_path)
    with open(model_out_path + 'args.txt', 'w') as f:
        for arg in vars(opt):
            print('%s: %s' % (arg, getattr(opt, arg)), file=f)
    joblib.dump(opt, model_out_path + 'args.pkl')
    train_dataset = FD(is_crop=True, split='train', crop_size=opt.train_img, MRI_path=MRI_PATH, vi_path=Other_PATH)
    train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True, pin_memory=True, drop_last=True)
    print("the training dataset is length:{}".format(train_dataset.length))
    # 导入模型
    model = select_model(opt.model_name)  # image_vis, image_ir
    model = model.cuda()
    print(model)
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=opt.betas, eps=opt.eps, weight_decay=0)
    # 损失函数
    criterion = select_loss(opt.loss_type)  # image_A, image_B, image_fused
    # 训练
    print('===> Starting training')
    tic = time.time()
    for epoch in range(opt.epochs):
        total_loss, losses_gdt, losses_l1, losses_SSIM = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        model.train()
        start = time.time()
        for current_step, (train_data, file_names) in enumerate(train_loader):
            # img_cat = train_data['img'].cuda()
            MRI_img = train_data['MRI'].cuda()
            Other_img = train_data['Other'].cuda()
            img_cat = torch.cat((Other_img, MRI_img), 1).cuda()

            output = model(Other_img, MRI_img, img_cat)
            fusion_loss, loss_gradient, loss_l1, loss_SSIM = criterion(MRI_img, Other_img, output)

            total_loss.update(fusion_loss.item(), Other_img.size(0))
            losses_gdt.update(loss_gradient.item(), Other_img.size(0))
            losses_l1.update(loss_l1.item(), Other_img.size(0))
            losses_SSIM.update(loss_SSIM.item(), Other_img.size(0))

            optimizer.zero_grad()
            # fusion_loss.requires_grad_(True)
            fusion_loss.backward()
            optimizer.step()

        train_log = f'Epoch {epoch}/{opt.epochs} : total_loss: {total_loss.avg:.4f}  L1_loss: {losses_l1.avg:.04f}  gd_Loss: {losses_gdt.avg:.04f}  ' \
                    f'SSIM_losses: {losses_SSIM.avg:.04f}  time:{time_to_str((time.time() - start), "sec")}'
        print(train_log)
        with open(model_out_path + 'args.txt', 'w') as f:
            print(train_log, file=f)

        if (epoch + 1) % opt.print_seq == 0 and (epoch+1 >= 500):
            torch.save(model.state_dict(), model_out_path + f'model_{epoch + 1}.pth')

    print(f"training is finish! total train time: {time_to_str((time.time() - tic), 'sec')}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--status', type=str, default='train', help='train or test')
    parser.add_argument('--model_name', type=str, default='DMFNet_UNet_skl4', help='MODEL NAME')
    parser.add_argument('--loss_type', default='DMFNet_loss', type=str, help='DMFNet_loss')
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--print_seq', default=100, type=int)
    parser.add_argument('--train_img', default=64, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', '--learning-rate', default=2e-4, type=float)
    parser.add_argument('--weight', default=[1, 1, 0.0005, 0.00056], type=float)
    parser.add_argument('--betas', default=(0.9, 0.999), type=tuple)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--eva', default=True, help='True or False')  # 是否评估
    opt = parser.parse_args()

    localtime = time.strftime("%Y%m%d_%H%M", time.localtime())  # 日期_时分，20230816_2351
    # 获得数据集
    
    train_name = 'train5'  # train5 train_functional
    root_dir = "/media/sata1/hedan/KTZ_DATA/"
    MRI_PATH = root_dir + f'{train_name}/pair1'
    Other_PATH = root_dir + f'{train_name}/pair2'
    print(localtime)
    data_root = f'/media/sata1/hedan/test_imgs_IN_JPG'
    run_train()