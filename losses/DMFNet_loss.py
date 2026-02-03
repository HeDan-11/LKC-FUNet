import torch
import torch.nn as nn
import torch.nn.functional as F
# Parts of these codes are from: https://github.com/Linfeng-Tang/SeAFusion
from torch.autograd import Variable
import numpy as np
import os
from math import exp
from losses.SwinF_ssim import ssim

class L_SSIM(nn.Module):
    def __init__(self):
        super(L_SSIM, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self, image_A, image_B, image_fused):
        Loss_SSIM = 0.5 * ssim(image_A, image_fused) + 0.5 * ssim(image_B, image_fused)
        return Loss_SSIM
    

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

    def forward(self, x):
        b, c, w, h = x.shape
        batch_list = []
        for i in range(b):
            tensor_list = []
            for j in range(c):
                sobelx_0 = F.conv2d(torch.unsqueeze(torch.unsqueeze(x[i, j, :, :], 0), 0), self.weightx, padding=1)
                sobely_0 = F.conv2d(torch.unsqueeze(torch.unsqueeze(x[i, j, :, :], 0), 0), self.weighty, padding=1)
                add_0 = torch.abs(sobelx_0) + torch.abs(sobely_0)
                tensor_list.append(add_0)

            batch_list.append(torch.stack(tensor_list, dim=1))

        return torch.cat(batch_list, dim=0)


def gradient(input):
    filter1 = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
    filter2 = torch.Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)
    Gradient1 =F.conv2d(input, filter1.to(input.get_device()), bias=None, stride=1, padding=1, dilation=1, groups=1)
    Gradient2 = F.conv2d(input, filter2.to(input.get_device()), bias=None, stride=1, padding=1, dilation=1, groups=1)
    Gradient = torch.abs(Gradient1) + torch.abs(Gradient2)
    return Gradient


class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv = Sobelxy()
        self.mse_criterion = torch.nn.MSELoss(reduce=True, size_average=True)
        self.L_SSIM = L_SSIM()

    def forward(self, MRI, OTHER, generate_img):
        image_vis, image_ir = OTHER, MRI
        image_y = image_vis

        # 梯度损失，基于融合图像
        out_grad = torch.mean(gradient(generate_img), dim=[1, 2, 3])
        grad_loss = 1 - torch.mean(out_grad / (out_grad + 1.0))

        B, C, W, H = image_vis.shape
        image_ir = image_ir.expand(B, C, W, H)
        x_in_max = torch.maximum(image_y, image_ir)
        loss_in = F.l1_loss(generate_img, x_in_max)
        # loss_in = self.mse_criterion(generate_img, x_in_max)

        # loss_in = loss_in + F.l1_loss(generate_img, image_y) + F.l1_loss(generate_img, image_ir)
        # Gradient
        y_grad = self.sobelconv(image_y)
        ir_grad = self.sobelconv(image_ir)
        B, C, K, W, H = y_grad.shape
        ir_grad = ir_grad.expand(B, C, K, W, H)
        generate_img_grad = self.sobelconv(generate_img)
        x_grad_joint = torch.maximum(y_grad, ir_grad)
        # loss_grad = F.l1_loss(generate_img_grad, x_grad_joint) + 0.5 * grad_loss
        loss_grad = F.l1_loss(generate_img_grad, x_grad_joint)
        # loss_grad = F.l1_loss(generate_img_grad, x_grad_joint) + 2 * grad_loss
        # ssim + grad
        # loss_grad = (1 - ssim(generate_img, image_ir, image_vis)) + loss_grad
        # loss_grad = self.mse_criterion(generate_img, image_vis) + self.mse_criterion(generate_img, image_ir)
        # ssim
        loss_SSIM = (1 - self.L_SSIM(MRI, OTHER, generate_img))


        # fusion_loss = 1.0*loss_in + 1.0*loss_grad + loss_SSIM
        fusion_loss = 1.0*loss_in + 1.0*loss_SSIM + loss_grad
        # fusion_loss = 1.0*loss_in + 1.0*loss_grad

        # loss_grad = ssim_raw(generate_img,image_y)
        return fusion_loss, loss_grad, loss_in, loss_SSIM