
import numpy  as np
import os
import cv2
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from util import parse_car_csv


class CustomFinetuneDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        #把所有图片读取进来
        # jpeg_images=[]
        # for sample_name in samples:
        #     dir=os.path.join(root_dir, 'JPEGImages', sample_name + ".jpg")
        #     s=cv2.imread(dir)
        #     jpeg_images.append(s)
        #读取目录
        samples = parse_car_csv(root_dir)
        #
        jpeg_images = [cv2.imread(os.path.join(root_dir, 'JPEGImages', sample_name + ".jpg"))
                       for sample_name in samples]

        positive_annotations = [os.path.join(root_dir, 'Annotations', sample_name + '_1.csv')
                                for sample_name in samples]
        negative_annotations = [os.path.join(root_dir, 'Annotations', sample_name + '_0.csv')
                                for sample_name in samples]

        # 边界框大小
        positive_sizes = []
        negative_sizes = []
        # 边界框坐标
        positive_rects = []
        negative_rects = []

        for annotation_path in positive_annotations:
            rects = np.loadtxt(annotation_path, dtype=np.int, delimiter=' ')
            # 存在文件为空或者文件中仅有单行数据
            if len(rects.shape) == 1:
                # 是否为单行
                if rects.shape[0] == 4:
                    positive_rects.append(rects)
                    positive_sizes.append(1)
                else:
                    positive_sizes.append(0)
            else:
                positive_rects.extend(rects)
                positive_sizes.append(len(rects))

        for annotation_path in negative_annotations:
            rects = np.loadtxt(annotation_path, dtype=np.int, delimiter=' ')
            # 和正样本规则一样
            if len(rects.shape) == 1:
                if rects.shape[0] == 4:
                    negative_rects.append(rects)
                    negative_sizes.append(1)
                else:
                    positive_sizes.append(0)
            else:
                negative_rects.extend(rects)
                negative_sizes.append(len(rects))

        self.transform = transform
        self.jpeg_images = jpeg_images
        self.positive_sizes = positive_sizes
        self.negative_sizes = negative_sizes
        self.positive_rects = positive_rects
        self.negative_rects = negative_rects
        self.total_positive_num = int(np.sum(positive_sizes))
        self.total_negative_num = int(np.sum(negative_sizes))

    def __getitem__(self, index: int):
        # 定位下标所属图像
        image_id = len(self.jpeg_images) - 1
        if index < self.total_positive_num:
            # 正样本
            target = 1
            xmin, ymin, xmax, ymax = self.positive_rects[index]
            # 寻找所属图像
            for i in range(len(self.positive_sizes) - 1):
                if np.sum(self.positive_sizes[:i]) <= index < np.sum(self.positive_sizes[:(i + 1)]):
                    image_id = i
                    break
            image = self.jpeg_images[image_id][ymin:ymax, xmin:xmax]
        else:
            # 负样本
            target = 0
            idx = index - self.total_positive_num
            xmin, ymin, xmax, ymax = self.negative_rects[idx]
            # 寻找所属图像
            for i in range(len(self.negative_sizes) - 1):
                if np.sum(self.negative_sizes[:i]) <= idx < np.sum(self.negative_sizes[:(i + 1)]):
                    image_id = i
                    break
            image = self.jpeg_images[image_id][ymin:ymax, xmin:xmax]

        if self.transform:
            image = self.transform(image)

        return image, target

    def __len__(self):
        return self.total_positive_num + self.total_negative_num

    def get_positive_num(self):
        return self.total_positive_num

    def get_negative_num(self):
        return self.total_negative_num




