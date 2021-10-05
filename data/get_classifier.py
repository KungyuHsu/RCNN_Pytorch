import shutil
import time

import cv2
from util import SelectiveSearch
from util import parse_xml
from util import compute_ious
import numpy as np
import os
#载入图片并selectivesearch选出来加入文件
dst_root="./classifier/"
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

    for i,iou in enumerate(iou_list):
        xmin,ymin,xmax,ymax=rects[i]
        size=(xmax-xmin)*(ymax-ymin)
        if iou>0.3:
            positive_list.append(rects[i])
        elif iou>0 and size>maxare/5.0:
            negative_list.append(rects[i])

    return bbs,negative_list

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

        samples = np.loadtxt(src_csv,dtype=np.str_)
#        shutil.copyfile(src_csv,dst_csv)
        tlt=len(samples)
        car = []
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
        np.savetxt(dst_csv, np.asarray(car),fmt='%s',delimiter=" ")
        print('%s positive num: %d' % (name, positive_number))
        print('%s negative num: %d' % (name, negative_number))
    print('done')
