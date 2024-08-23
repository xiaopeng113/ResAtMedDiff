import os

import cv2
import numpy
import torch
from torch.utils.data import Dataset


class Get_tager_sample_seg(Dataset,):
    def __init__(self, path,):
        self.path = path
        self.name = os.listdir(path)

    def __getitem__(self, index):
        path_img_root = self.name[index]
        root_img  = os.path.join( self.path,path_img_root)
        flair_path  = os.path.join(root_img,path_img_root+"_flair.gz")
        seg_path = os.path.join(root_img, path_img_root + "_seg.gz")
        t1_path = os.path.join(root_img, path_img_root + "_t1.gz")
        t1ce_path = os.path.join(root_img, path_img_root + "_t1ce.gz")
        t2_path = os.path.join(root_img, path_img_root + "_t2.gz")
        # print(seg)
        # print(path_img_root)
        flair_50 = cv2.imread(os.path.join(flair_path,"50.png"))[...,0]
        flair_50 = cv2.resize(flair_50, (256, 256))
        seg_50 = cv2.imread(os.path.join(seg_path,"50.png"))[...,0]
        seg_50 = cv2.resize(seg_50, (256, 256))
        t1_50 = cv2.imread(os.path.join(t1_path,"50.png"))[...,0]
        t1_50 = cv2.resize(t1_50, (256, 256))
        t1ce_50 = cv2.imread(os.path.join(t1ce_path, "50.png"))[...,0]
        t1ce_50 = cv2.resize(t1ce_50, (256, 256))
        t2_50 = cv2.imread(os.path.join(t2_path, "50.png"))[...,0]
        t2_50 = cv2.resize(t2_50, (256, 256))

        # end = t1_50+t1ce_50+t2_50+flair_50
        end = numpy.stack([t1_50, t1ce_50,t2_50,flair_50],axis=0)


        seg_50 = cv2.resize(seg_50,(256,256))
        seg_50 = torch.unsqueeze(torch.from_numpy(seg_50),dim=0).numpy()

        return end.astype(numpy.float32) ,(seg_50).astype(numpy.float32)


    def __len__(self):
        return len(self.name)

data = Get_tager_sample_seg("/data/brats/")
data_ = torch.utils.data.DataLoader(data,batch_size=350)
t1,t2=next(iter(data_))
print(t1.mean(),t1.std(),t2.mean(),t2.std())

