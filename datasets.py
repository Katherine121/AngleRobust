import os
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import imgaug.augmenters as iaa


class TrainDataset(Dataset):
    def __init__(self, dataset_path, num_nodes, transform, input_len, add_cont_style):
        """
        train dataset, form a sequence every five frames with an end point frame.
        :param transform: torchvision.transforms.
        :param input_len: input sequence length (not containing the end point).
        """
        self.transform = transform
        self.input_len = input_len
        self.add_cont_style = add_cont_style

        res_seq = []

        # 一个file是一条有100张左右图像的路径
        for i in range(0, num_nodes, 5):
            for j in range(i + 1, num_nodes, 5):
                for k in range(0, 20):
                    # 0.8 as train dataset
                    # 0-15
                    if k <= 15:
                        path_seq = []
                        full_file_path = os.path.join(dataset_path,
                                                      "path" + str(i) + "," + str(j) + "," + str(k) + ".txt")
                        f = open(full_file_path, 'rt')
                        for line in f:
                            line = line.strip('\n')
                            line = line.split(' ')

                            # read frame, angle,
                            # end point frame,
                            # the current position label,
                            # the next position label, the direction angle
                            line = [line[0], [float(line[1]), float(line[2])],
                                    line[3],
                                    int(line[4]),
                                    int(line[5]), [float(line[6]), float(line[7])]
                                    ]
                            path_seq.append(line)
                        f.close()

                        # if there are not enough input frames
                        for index in range(1, len(path_seq) + 1):
                            if index - self.input_len < 0:
                                res_seq.append(path_seq[0: index])
                            else:
                                res_seq.append(path_seq[index - self.input_len: index])

        print(len(res_seq))
        self.imgs = res_seq

    def __len__(self):
        """
        return the length of the dataset.
        :return:
        """
        return len(self.imgs)

    def __getitem__(self, index):
        """
        read the image sequence, angle sequence and label corresponding to the index in the dataset.
        :param index: index of self.imgs.
        :return: frame sequence, angle sequence,
                the current position label, the next position label, the direction angle.
        """
        if self.add_cont_style:
            style_idx = random.randint(0, 4)
            if style_idx == 0:
                style_transform = iaa.imgcorruptlike.Brightness()
            elif style_idx == 1:
                style_transform = iaa.Rain()
            elif style_idx == 2:
                style_transform = iaa.imgcorruptlike.Snow()
            elif style_idx == 3:
                style_transform = iaa.imgcorruptlike.Fog()
            elif style_idx == 4:
                style_transform = iaa.Cutout(size=random.randint(1, 3) * 0.1)

        item = self.imgs[index]

        next_imgs = None
        cont_next_imgs = None
        next_angles = []

        # generate frame sequence and angle sequence
        for i in range(0, len(item)):
            img = item[i][0]
            img = Image.open(img)
            img = img.convert('RGB')

            if self.add_cont_style:
                cont_img = np.array(img)
                cont_img = style_transform(image=cont_img)
                cont_img = Image.fromarray(cont_img)

            img = self.transform(img).unsqueeze(dim=0)
            cont_img = self.transform(cont_img).unsqueeze(dim=0)

            if next_imgs is None:
                next_imgs = img
                cont_next_imgs = cont_img
            else:
                next_imgs = torch.cat((next_imgs, img), dim=0)
                cont_next_imgs = torch.cat((cont_next_imgs, cont_img), dim=0)

            if i == len(item) - 1:
                next_angles.append([0, 0])
            else:
                next_angles.append(item[i][1])

        # append the end point frame as part of model input
        dest_img = Image.open(item[-1][2])
        dest_img = dest_img.convert('RGB')
        dest_img = self.transform(dest_img).unsqueeze(dim=0)
        next_imgs = torch.cat((next_imgs, dest_img), dim=0)
        cont_next_imgs = torch.cat((cont_next_imgs, dest_img), dim=0)

        dest_angle = [0, 0]
        next_angles.append(dest_angle)
        next_angles = torch.tensor(next_angles, dtype=torch.float)

        # the current position label, the next position label, the direction angle
        label1 = item[-1][3]
        label2 = item[-1][4]
        label3 = torch.tensor(item[-1][5], dtype=torch.float)

        # if there are not enough input frames
        for i in range(0, self.input_len - len(item)):
            next_imgs = torch.cat((next_imgs, torch.zeros((1, 3, 224, 224))), dim=0)
            cont_next_imgs = torch.cat((cont_next_imgs, torch.zeros((1, 3, 224, 224))), dim=0)
            next_angles = torch.cat((next_angles, torch.zeros((1, 2))), dim=0)

        # input: frame sequence (input_len + 1), angle sequence (input_len + 1),
        # output: the current position label, the next position label, the direction angle
        return next_imgs, cont_next_imgs, next_angles, label1, label2, label3


class TestDataset(Dataset):
    def __init__(self, dataset_path, num_nodes, transform, input_len, cont_style):
        """
        test dataset, form a sequence every five frames with an end point frame.
        :param transform: torchvision.transforms.
        :param input_len: input sequence length (not containing the end point).
        """
        self.transform = transform
        self.input_len = input_len
        self.cont_style = cont_style

        res_seq = []

        # 一个file是一条有100张左右图像的路径
        for i in range(0, num_nodes, 5):
            for j in range(i + 1, num_nodes, 5):
                for k in range(0, 20):
                    # 0.8 as train dataset
                    # 16-19
                    if k >= 16:
                        path_seq = []
                        full_file_path = os.path.join(dataset_path,
                                                      "path" + str(i) + "," + str(j) + "," + str(k) + ".txt")
                        f = open(full_file_path, 'rt')
                        for line in f:
                            line = line.strip('\n')
                            line = line.split(' ')

                            # read frame, angle,
                            # end point frame,
                            # the current position label,
                            # the next position label, the direction angle.
                            line = [line[0], [float(line[1]), float(line[2])],
                                    line[3],
                                    int(line[4]),
                                    int(line[5]), [float(line[6]), float(line[7])]
                                    ]
                            path_seq.append(line)
                        f.close()

                        # if there are not enough input frames
                        for index in range(1, len(path_seq) + 1):
                            if index - self.input_len < 0:
                                res_seq.append(path_seq[0: index])
                            else:
                                res_seq.append(path_seq[index - self.input_len: index])

        print(len(res_seq))
        self.imgs = res_seq

    def __len__(self):
        """
        return the length of the dataset.
        :return:
        """
        return len(self.imgs)

    def __getitem__(self, index):
        """
        read the image sequence, angle sequence and label corresponding to the index in the dataset.
        :param index: index of self.imgs.
        :return: frame sequence, angle sequence,
                the current position label, the next position label, the direction angle.
        """
        if self.cont_style:
            style_idx = random.randint(0, 4)
            if style_idx == 0:
                style_transform = iaa.imgcorruptlike.Brightness()
            elif style_idx == 1:
                style_transform = iaa.Rain()
            elif style_idx == 2:
                style_transform = iaa.imgcorruptlike.Snow()
            elif style_idx == 3:
                style_transform = iaa.imgcorruptlike.Fog()
            elif style_idx == 4:
                style_transform = iaa.Cutout(size=random.randint(1, 3) * 0.1)

        item = self.imgs[index]

        next_imgs = None
        next_angles = []

        # generate frame sequence and angle sequence
        for i in range(0, len(item)):
            img = item[i][0]
            img = Image.open(img)
            img = img.convert('RGB')

            if self.cont_style:
                img = np.array(img)
                img = style_transform(image=img)
                img = Image.fromarray(img)

            img = self.transform(img).unsqueeze(dim=0)

            if next_imgs is None:
                next_imgs = img
            else:
                next_imgs = torch.cat((next_imgs, img), dim=0)

            if i == len(item) - 1:
                next_angles.append([0, 0])
            else:
                next_angles.append(item[i][1])

        # append the end point frame as part of model input
        dest_img = Image.open(item[-1][2])
        dest_img = dest_img.convert('RGB')
        dest_img = self.transform(dest_img).unsqueeze(dim=0)
        next_imgs = torch.cat((next_imgs, dest_img), dim=0)

        dest_angle = [0, 0]
        next_angles.append(dest_angle)
        next_angles = torch.tensor(next_angles, dtype=torch.float)

        # the current position label, the next position label, the direction angle
        label1 = item[-1][3]
        label2 = item[-1][4]
        label3 = torch.tensor(item[-1][5], dtype=torch.float)

        # if there are not enough input frames
        for i in range(0, self.input_len - len(item)):
            next_imgs = torch.cat((next_imgs, torch.zeros((1, 3, 224, 224))), dim=0)
            next_angles = torch.cat((next_angles, torch.zeros((1, 2))), dim=0)

        # input: frame sequence (input_len + 1), angle sequence (input_len + 1),
        # output: the current position label, the next position label, the direction angle
        return next_imgs, next_angles, label1, label2, label3
