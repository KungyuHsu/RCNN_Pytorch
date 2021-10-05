import shutil

from torchvision.datasets import VOCDetection
import pandas as pd
import numpy as np
import os
#下载数据
#dataset = VOCDetection('./data', year='2007', image_set='trainval', download=True)

kinds = ["car"]

src = "VOCdevkit/VOC2007"
Main_dir = "./VOCdevkit/VOC2007/ImageSets/Main"
tar_dit = "./deal_data"

def get_data(dir,kinds,type):
    """
    把对应正样本提取出来
    :param dir: imageSets的Main路径
    :param kinds: 提取几种东西的名称
    :param type: 提取的批次名称，train还是val还是trainval
    :return:返回[kinds,number]的list
    """

    #res = []
    # print(data)
    # with open(dir_kind_t,'r') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         sp = line.split(" ")
    #         if len(res)==3 and res[2]==1:
    #             res.append(sp[0])
    # tlt.append(res)
    #res=[]
    tlt= []
    for kind in kinds:
        dir_kind_t = dir+"/"+kind+"_"+type+".txt"
        data = pd.read_csv(dir_kind_t,sep=" ",header=0,names=[0,1],index_col=False,dtype=str)
        res = np.asarray(data[data[1].isnull()== True][0])
        tlt.append(res)
    return tlt
#拷贝标注样本图片以及xml文件
def copy_file(samples):
    for sample in ["train","val"]:
        data_list=samples[sample]

        for name in data_list[0]:
            XML_from = src+"/Annotations/"+name+".xml"
            XML_To = tar_dit+"/"+sample+"/Annotations/"+name+".xml"
            JPG_from = src+"/JPEGImages/"+name+".jpg"
            JPG_To = tar_dit+"/"+sample+"/JPEGImages/"+name+".jpg"
            shutil.copy(src=XML_from, dst=XML_To)
            shutil.copy(src=JPG_from, dst=JPG_To)
        data_csv = tar_dit + "/" + sample + "/car.csv"
        np.savetxt(data_csv,np.asarray(data_list),fmt="%s")
#读取分类文件
samples = {"train":get_data(Main_dir, kinds, "train"), "val":get_data(Main_dir, kinds, "val")}
copy_file(samples)
print("="*10+"Copy done!"+"="*10)