import shutil
import time

import cv2
from util import SelectiveSearch
from util import parse_xml
from util import compute_ious
import numpy as np
import os
#载入图片并selectivesearch选出来加入文件
dst_root="./funetune/"
src_root="./deal_data/"
selectivesearch = SelectiveSearch(mode="f")
def get_list(src_xml,src_jpg):
    img = cv2.imread(src_jpg)
    #Selective Search
    rects = selectivesearch(img)
    #标注文件
    bbs=parse_xml(src_xml)

    if len(bbs)>0:
        iou_list=compute_ious(rects,bbs)
    else:
        return [],[]

    maxare=-1
    for bb in bbs:
        xmin,ymin,xmax,ymax=bb
        are = (xmax-xmin)*(ymax-ymin)
        if are>maxare:
            maxare=are

    positive_list=[]
    negative_list=[]

    #分类判断
    for i,iou in enumerate(iou_list):
        #获取坐标、计算大小
        xmin,ymin,xmax,ymax=rects[i]
        size=(xmax-xmin)*(ymax-ymin)
        if iou>0.5:#正样本
            positive_list.append(rects[i])
        elif iou>0 and size>maxare/5.0:
            #负样本
            negative_list.append(rects[i])

    return positive_list,negative_list

if __name__ == '__main__':
    for name in ['train','val']:
        positive_number=0
        negative_number=0

        src=os.path.join(src_root,name)
        dst=os.path.join(dst_root,name)
        dst_csv=os.path.join(dst,"car.csv")
        dst_annotation = os.path.join(dst,"Annotations")
        dst_jpeg = os.path.join(dst,"JPEGImages")
        src_csv=os.path.join(src,"car.csv")
        src_annotations = os.path.join(src, "Annotations")
        src_img = os.path.join(src, "JPEGImages")

        #读取目录
        samples = np.loadtxt(src_csv,dtype=np.str_)
        tlt=len(samples)
        car = []
        #将每个图片输入到SelectiveSearch中
        #将IOU>0.5设置为正样本
        for i,sample in enumerate(samples):
            since = time.time()
            src_xml=os.path.join(src_annotations,sample+".xml")
            src_jpg=os.path.join(src_img,sample+".jpg")

            positive_list,negative_list = get_list(src_xml,src_jpg)
            if len(positive_list)==0:
                continue
            car.append(sample)
            positive_number+=len(positive_list)
            negative_number+=len(negative_list)

            dst_annotation_positive_path = os.path.join(dst_annotation, sample + '_1' + '.csv')
            dst_annotation_negative_path = os.path.join(dst_annotation, sample + '_0' + '.csv')
            dst_jpeg_path = os.path.join(dst_jpeg, sample + '.jpg')
            # 保存图片
            shutil.copyfile(src_jpg, dst_jpeg_path)
            # 保存正负样本标注
            np.savetxt(dst_annotation_positive_path, np.array(positive_list), fmt='%d', delimiter=' ')
            np.savetxt(dst_annotation_negative_path, np.array(negative_list), fmt='%d', delimiter=' ')
            time_elapsed = time.time() - since
            print('{}parse {}.png in {:.0f}m {:.0f}s process: [{}/{}] {:.2f}%'.format(name,sample, time_elapsed // 60, time_elapsed % 60,i,tlt,(i/tlt)*100))
        #保存提取出来的编号列表
        np.savetxt(dst_csv, np.asarray(car),fmt='%s',delimiter=" ")

        print('%s positive num: %d' % (name, positive_number))
        print('%s negative num: %d' % (name, negative_number))
    print('done')
