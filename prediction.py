import time

import torch

from selective_search import SelectiveSearch
import cv2
from torchvision.models import AlexNet
import torchvision.transforms as transforms
import util
import numpy as np
def get_model(device=None):
    # 加载CNN模型
    model = AlexNet(num_classes=2)
    model.load_state_dict(torch.load('alexnet_car_classier.pth',map_location=torch.device('cpu')))
    model.eval()

    # 取消梯度追踪
    for param in model.parameters():
        param.requires_grad = False
    if device:
        model = model.to(device)

    return model
def get_bbr(device=None):
    # 加载CNN模型
    bbr=torch.load('bbox_regression_MLP36_5003.pth',map_location=torch.device('cpu'))
    bbr.eval()

    # 取消梯度追踪
    for param in bbr.parameters():
        param.requires_grad = False
    if device:
        bbr = bbr.to(device)
    return bbr

def nms(rect_list, score_list):
    """
    非最大抑制
    :param rect_list: list，大小为[N, 4]
    :param score_list： list，大小为[N]
    """
    nms_rects = list()
    nms_scores = list()

    rect_array = np.array(rect_list)
    score_array = np.array(score_list)

    # 一次排序后即可
    # 按分类概率从大到小排序
    idxs = np.argsort(score_array)[::-1]
    rect_array = rect_array[idxs]
    score_array = score_array[idxs]

    thresh = 0.3
    while len(score_array) > 0:
        # 添加分类概率最大的边界框
        nms_rects.append(rect_array[0])
        nms_scores.append(score_array[0])
        rect_array = rect_array[1:]
        score_array = score_array[1:]

        length = len(score_array)
        if length <= 0:
            break

        # 计算IoU
        iou_scores = util.iou(np.array(nms_rects[len(nms_rects) - 1]), rect_array)
        # print(iou_scores)
        # 去除重叠率大于等于thresh的边界框
        idxs = np.where(iou_scores < thresh)[0]
        rect_array = rect_array[idxs]
        score_array = score_array[idxs]

    return nms_rects, nms_scores
class BBR:
    def __init__(self,alexnet,bbr):
        self.feaure = alexnet.features
        self.bodinbox = bbr
    def __call__(self,rect,img):
        xmin, ymin, xmax, ymax = rect
        transimg=transform(img[ymin:ymax, xmin:xmax])
        feat = self.feaure(transimg.unsqueeze(0))
        feat = torch.flatten(feat, 1)
        d_x, d_y,d_w, d_h=self.bodinbox(feat)[0]
        p_w = xmax - xmin
        p_h = ymax - ymin
        p_x = xmin + p_w / 2
        p_y = ymin + p_h / 2
        Gx = p_w*d_x + p_x
        Gy = p_h*d_y + p_y
        Gw = p_w*np.exp(d_w)
        Gh = p_h*np.exp(d_h)
        Gxmin = Gx - Gw/2
        Gymin = Gy - Gh/2
        Gxmax = Gxmin + Gw
        Gymax = Gymin + Gh
        return Gxmin,Gymin,Gxmax,Gymax
if __name__ == '__main__':
    #定义SelectiveSearch算法
    ss=SelectiveSearch()
    #加载模型
    model = get_model()
    bb = get_bbr()
    bbr = BBR(model, bb)
    #定义转换函数
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((227, 227)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    #定义SVM阙值
    SVM_Gate=0.6
    start = time.time()
    print("=====start process=====")
    #加载图片
    img = cv2.imread("./data/test8.png")
    #进行Selective Search
    rects=ss(img)
    reslist=[]
    score=[]
    #向前传播
    for id,rect in enumerate(rects):
        xmin,ymin,xmax,ymax = rect
        #截取图片并转换
        regimg = img[ymin:ymax,xmin:xmax]
        transimg=transform(regimg)
        #向前传播
        res = model(transimg.unsqueeze(0))
        #根据阙值筛选
        if res[0][1]>0.6:
            score.append(res[0][1])
            reslist.append(rect)
    #进行最大值抑制
    nms_rects, nms_scores = nms(reslist, score)

    end = time.time()-start
    print("using time:",end)

    #回归框精修
    for rect in nms_rects:
        #画原来的框
        xmin, ymin, xmax, ymax = rect
        cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,0,255),2)
        #向前传播
        xmin, ymin, xmax, ymax = bbr(rect,img)
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        #画回归后的框（绿色）
        cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,255,0),2)

    print("tlt",nms_rects.__len__())
    cv2.imshow("testimg",img)
    k=cv2.waitKey(0)
    if k==27:
        cv2.destroyWindow("testimg")