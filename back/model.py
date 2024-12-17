import numpy as np
from skimage import io, measure
import random
import matplotlib.pyplot as plt
from preclassify import dicomp, hcluster
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import cv2
import os

im1_path  = 'ottawa_1.bmp'
im2_path  = 'ottawa_2.bmp'
imgt_path = 'ottawa_gt.bmp'

# important parameter
patch_size = 7

def image_normalize(data):
    import math
    _mean = np.mean(data)
    _std = np.std(data)
    npixel = np.size(data) * 1.0
    min_stddev = 1.0 / math.sqrt(npixel)
    return (data - _mean) / max(_std, min_stddev)


def image_padding(data, r):
    if len(data.shape) == 3:
        data_new = np.lib.pad(data, ((r, r), (r, r), (0, 0)), 'constant', constant_values=0)
        return data_new
    if len(data.shape) == 2:
        data_new = np.lib.pad(data, r, 'constant', constant_values=0)
        return data_new


# 生成自然数数组并打乱
def arr(length):
    arr = np.arange(length - 1)
    # print(arr)
    random.shuffle(arr)
    # print(arr)
    return arr


# 在每个像素周围提取 patch ，然后创建成符合 pytorch 处理的格式
def createTrainingCubes(X, y, patch_size):
    # 给 X 做 padding
    margin = int((patch_size - 1) / 2)
    zeroPaddedX = image_padding(X, margin)
    # 把类别 uncertainty 的像素忽略
    ele_num1 = np.sum(y == 1)
    ele_num2 = np.sum(y == 2)
    patchesData_1 = np.zeros((ele_num1, patch_size, patch_size, X.shape[2]))
    patchesLabels_1 = np.zeros(ele_num1)

    patchesData_2 = np.zeros((ele_num2, patch_size, patch_size, X.shape[2]))
    patchesLabels_2 = np.zeros(ele_num2)

    patchIndex_1 = 0
    patchIndex_2 = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            # remove uncertainty pixels
            if y[r - margin, c - margin] == 1:
                patch_1 = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
                patchesData_1[patchIndex_1, :, :, :] = patch_1
                patchesLabels_1[patchIndex_1] = y[r - margin, c - margin]
                patchIndex_1 = patchIndex_1 + 1
            elif y[r - margin, c - margin] == 2:
                patch_2 = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
                patchesData_2[patchIndex_2, :, :, :] = patch_2
                patchesLabels_2[patchIndex_2] = y[r - margin, c - margin]
                patchIndex_2 = patchIndex_2 + 1
    patchesLabels_1 = patchesLabels_1 - 1
    patchesLabels_2 = patchesLabels_2 - 1

    # 调用arr函数打乱数组
    arr_1 = arr(len(patchesData_1))
    arr_2 = arr(len(patchesData_2))

    train_len = 10000  # 设置训练集样本数
    pdata = np.zeros((train_len, patch_size, patch_size, X.shape[2]))
    plabels = np.zeros(train_len)

    for i in range(7000):
        pdata[i, :, :, :] = patchesData_1[arr_1[i], :, :, :]
        plabels[i] = patchesLabels_1[arr_1[i]]
    for j in range(7000, train_len):
        pdata[j, :, :, :] = patchesData_2[arr_2[j - 7000], :, :, :]
        plabels[j] = patchesLabels_2[arr_2[j - 7000]]

    return pdata, plabels


def createTestingCubes(X, patch_size):
    # 给 X 做 padding
    margin = int((patch_size - 1) / 2)
    zeroPaddedX = image_padding(X, margin)
    patchesData = np.zeros((X.shape[0] * X.shape[1], patch_size, patch_size, X.shape[2]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchIndex = patchIndex + 1
    return patchesData

def postprocess(res):
    res_new = res
    res = measure.label(res, connectivity=2)
    num = res.max()
    for i in range(1, num + 1):
        idy, idx = np.where(res == i)
        if len(idy) <= 20:
            res_new[idy, idx] = 0
    return res_new




""" Training dataset"""

class TrainDS(torch.utils.data.Dataset):
    def __init__(self):
        self.len = x_train.shape[0]
        self.x_data = torch.FloatTensor(x_train)
        self.y_data = torch.LongTensor(y_train)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签

        # x=torch.FloatTensor(data_rotate(self.x_data[index].cpu().numpy()))
        # y=torch.FloatTensor(gasuss_noise(self.y_data[index]))
        # x=torch.FloatTensor(datarotate(self.x_data[index]))
        # return x,self.y_data[index]
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        # 返回文件数据的数目
        return self.len


class MRC(nn.Module):
    def __init__(self, inchannel):
        super(MRC, self).__init__()
        self.conv1 = nn.Conv2d(inchannel, 15, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(15)

        self.conv2_1 = nn.Conv2d(5, 5, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2_1 = nn.BatchNorm2d(5)

        self.conv2_2 = nn.Conv2d(5, 5, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2_2 = nn.BatchNorm2d(5)

        self.conv2_3 = nn.Conv2d(5, 5, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2_3 = nn.BatchNorm2d(5)

    def forward(self, x):
        ori_out = F.relu(self.bn1(self.conv1(x)))

        shape = (x.shape[0], 5, 7, 7)

        all_zero3_3 = torch.zeros(size=shape).cuda()
        all_zero1_3 = torch.zeros(size=(x.shape[0], 5, 3, 7)).cuda()
        all_zero3_1 = torch.zeros(size=(x.shape[0], 5, 7, 3)).cuda()

        all_zero3_3[:, :, :, :] = ori_out[:, 0:5, :, :]
        all_zero1_3[:, :, :, :] = ori_out[:, 5:10, 2:5, :]
        all_zero3_1[:, :, :, :] = ori_out[:, 10:15, :, 2:5]

        square = F.relu(self.bn2_1(self.conv2_1(all_zero3_3)))
        horizontal = F.relu(self.bn2_2(self.conv2_2(all_zero1_3)))
        vertical = F.relu(self.bn2_3(self.conv2_3(all_zero3_1)))
        horizontal_final = torch.zeros(size=(x.shape[0], 5, 7, 7)).cuda()
        vertical_final = torch.zeros(size=(x.shape[0], 5, 7, 7)).cuda()
        horizontal_final[:, :, 2:5, :] = horizontal[:, :, :, :]
        vertical_final[:, :, :, 2:5] = vertical[:, :, :, :]

        glo = square + horizontal_final + vertical_final
        # glo= F.relu(self.bn3(self.conv3(glo)))

        return glo


def DCT(x):
    out = F.interpolate(x, size=(8, 8), mode='bilinear', align_corners=True)
    # print(out.shape)
    # dct_out_1 =torch.Tensor([cv2.dct(x[i,0,:,:].detach().cpu().numpy()) \
    #                          for i in range(x.shape[0])])
    dct_out_1 = torch.Tensor([cv2.dct(np.float32(out[i, 0, :, :].detach().cpu().numpy())) \
                              for i in range(x.shape[0])])
    dct_out_2 = torch.Tensor([cv2.dct(np.float32(out[i, 1, :, :].detach().cpu().numpy())) \
                              for i in range(x.shape[0])])
    dct_out_3 = torch.Tensor([cv2.dct(np.float32(out[i, 2, :, :].detach().cpu().numpy())) \
                              for i in range(x.shape[0])])
    dct_out = torch.zeros(size=(x.shape[0], 3, 8, 8))
    dct_out[:, 0, :, :] = dct_out_1
    dct_out[:, 1, :, :] = dct_out_2
    dct_out[:, 2, :, :] = dct_out_3
    dct_out = dct_out.cuda()  # 放回cuda
    out = dct_out.view(x.shape[0], 3, 64)
    # out=torch.cat((out,out),2)
    out = F.glu(out, dim=-1)
    dct_out = out.view(x.shape[0], 1, 96)
    return dct_out


class DDNet(nn.Module):
    def __init__(self):
        super(DDNet, self).__init__()
        self.mrc1 = MRC(3)
        self.mrc2 = MRC(5)
        self.mrc3 = MRC(5)
        self.mrc4 = MRC(5)

        self.linear1 = nn.Linear(341, 10)
        self.linear2 = nn.Linear(10, 2)

    def forward(self, x):
        m_1 = self.mrc1(x)
        m_2 = self.mrc2(m_1)
        m_3 = self.mrc3(m_2)
        m_4 = self.mrc4(m_3)
        # glo= F.relu(self.bn(self.conv(m_4)))
        glo = m_4.view(x.shape[0], 1, 245)

        dct_out = DCT(x)

        out = torch.cat((glo, dct_out), 2)
        out = out.view(out.size(0), -1)
        # print(out.shape)
        out_1 = self.linear1(out)
        out = self.linear2(out_1)

        return out


def change_detection(image1_path, image2_path, output_path='change_detection_result.png'):
    """
    变化检测函数

    参数：
        image1_path: str，第一幅图片路径
        image2_path: str，第二幅图片路径
        gt_path: str，地面真值图像路径
        output_path: str，保存检测结果的路径（默认值为 'change_detection_result.png'）

    输出：
        显示变化检测结果并保存为指定路径的图片
    """
    # 读取图片并转换为浮点型
    im1 = io.imread(image1_path)[:, :, 0].astype(np.float32)
    im2 = io.imread(image2_path)[:, :, 0].astype(np.float32)

    # 计算差分图像
    im_di = dicomp(im1, im2)
    ylen, xlen = im_di.shape

    # 分层聚类预分类
    pix_vec = im_di.reshape([ylen * xlen, 1])
    preclassify_lab = hcluster(pix_vec, im_di)

    # 创建模型输入数据
    mdata = np.zeros([im1.shape[0], im1.shape[1], 3], dtype=np.float32)
    mdata[:, :, 0] = im1
    mdata[:, :, 1] = im2
    mdata[:, :, 2] = im_di

    x_test = createTestingCubes(mdata, patch_size)
    x_test = x_test.transpose(0, 3, 1, 2)

    # 使用模型进行逐像素分类预测
    outputs = np.zeros((ylen, xlen))
    for i in range(ylen):
        for j in range(xlen):
            if preclassify_lab[i, j] != 1.5:
                outputs[i, j] = preclassify_lab[i, j]
            else:
                img_patch = x_test[i * xlen + j, :, :, :]
                img_patch = img_patch.reshape(1, img_patch.shape[0], img_patch.shape[1], img_patch.shape[2])
                img_patch = torch.FloatTensor(img_patch).to(device)
                prediction = net(img_patch)
                prediction = np.argmax(prediction.detach().cpu().numpy(), axis=1)
                outputs[i, j] = prediction + 1

    outputs = outputs - 1

    # 后处理和评估
    res = outputs * 255
    res = postprocess(res)

    # 显示并保存图片
    plt.imshow(res, 'gray')
    plt.imsave(output_path, res, cmap='gray')
    print(f'变化检测结果已保存到: {output_path}')

    return output_path
