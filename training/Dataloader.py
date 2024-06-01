import os
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage.transform import resize
from torchvision import transforms
from PIL import Image
import csv

class CheXpertDataSet(Dataset):
    def __init__(self, data_PATH, transform = None, policy = "ones", args = None):
        super(CheXpertDataSet, self).__init__()
        self.image_names = []
        self.labels = []

        with open(data_PATH, "r") as f:
            csvReader = csv.reader(f)
            next(csvReader, None)
            for line in csvReader:
                # the first column is the path of the image
                image_name = line[0]
                # Don't take from column 0 to 4 as they are the path of the image, sex, age, frontal, AP/PA
                label = line[5:]
                
                for i in range(args.num_classes):
                    if label[i]:
                        a = float(label[i])
                        if a == 1:
                            label[i] = 1
                        elif a == -1:
                            if policy == "ones":
                                label[i] = 1
                            elif policy == "zeroes":
                                label[i] = 0
                            else:
                                label[i] = 0
                        else:
                            label[i] = 0
                    else:
                        label[i] = 0
                        
                self.image_names.append('./' + image_name)
                self.labels.append(label)

        self.image_names = self.image_names
        self.labels = self.labels
        self.transform = transform
        assert len(self.image_names) == len(self.labels)

    def __getitem__(self, index):
        """Take the index of item and returns the image and its labels"""
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)