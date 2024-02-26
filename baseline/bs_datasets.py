import os
import torch
from PIL import Image
from torch.utils.data import Dataset


class BSTrainDataset(Dataset):
    def __init__(self, dataset_path, class_path, transform):
        """
        :param dataset_path: UAV_AR368 path
        :param class_path: clustered UAV_AR368 path
        :param transform: torchvision.transforms
        """
        self.transform = transform
        self.data = []
        self.target = []

        dir_path1 = os.listdir(dataset_path)
        dir_path1.sort()

        class_path1 = os.listdir(class_path)
        class_path1.sort()

        for i in range(0, len(dir_path1)):
            if i % 5 != 0:
                dir1 = dir_path1[i]
                full_dir_path = os.path.join(dataset_path, dir1)

                file_path1 = os.listdir(full_dir_path)
                file_path1.sort()

                f = open(class_path + "/" + class_path1[i])
                cluster_labels = []
                for line in f:
                    line = line.strip('\n')
                    cluster_labels.append(int(line))

                for j in range(0, len(file_path1)):
                    file1 = file_path1[j]
                    full_file_path = os.path.join(full_dir_path, file1)
                    self.data.append(full_file_path)
                    self.target.append(cluster_labels[j])

        print(len(self.data))
        self.imgs = self.data

    def __len__(self):
        """
        return the length of the dataset
        :return:
        """
        return len(self.imgs)

    def __getitem__(self, index):
        """
        read the image
        :param index: index of self.imgs
        :return: input tensor and label
        """
        img = Image.open(self.imgs[index])
        img = img.convert('RGB')
        img = self.transform(img)
        return img, torch.tensor(self.target[index], dtype=torch.int64)


class BSTestDataset(Dataset):
    def __init__(self, dataset_path, class_path, transform):
        """
        :param dataset_path: UAV_AR368 path
        :param class_path: clustered UAV_AR368 path
        :param transform: torchvision.transforms
        """
        self.transform = transform
        self.data = []
        self.target = []

        dir_path1 = os.listdir(dataset_path)
        dir_path1.sort()

        class_path1 = os.listdir(class_path)
        class_path1.sort()

        for i in range(0, len(dir_path1)):
            if i % 5 == 0:
                dir1 = dir_path1[i]
                full_dir_path = os.path.join(dataset_path, dir1)

                file_path1 = os.listdir(full_dir_path)
                file_path1.sort()

                f = open(class_path + "/" + class_path1[i])
                cluster_labels = []
                for line in f:
                    line = line.strip('\n')
                    cluster_labels.append(int(line))

                for j in range(0, len(file_path1)):
                    file1 = file_path1[j]
                    full_file_path = os.path.join(full_dir_path, file1)
                    self.data.append(full_file_path)
                    self.target.append(cluster_labels[j])

        print(len(self.data))
        self.imgs = self.data

    def __len__(self):
        """
        return the length of the dataset
        :return:
        """
        return len(self.imgs)

    def __getitem__(self, index):
        """
        read the image
        :param index: index of self.imgs
        :return: input tensor and label
        """
        img = Image.open(self.imgs[index])
        img = img.convert('RGB')
        img = self.transform(img)
        return img, torch.tensor(self.target[index], dtype=torch.int64)
