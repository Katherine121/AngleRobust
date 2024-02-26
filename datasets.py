import os
import torch
from PIL import Image
from torch.utils.data import Dataset


class OrderTrainDataset(Dataset):
    def __init__(self, dataset_path, transform, input_len):
        """
        train dataset, form a sequence every five frames with an end point frame
        :param transform: torchvision.transforms
        :param input_len: input sequence length (not containing the end point)
        """
        self.transform = transform
        self.input_len = input_len

        res_seq = []
        files = os.listdir(dataset_path)
        files.sort()
        for i in range(0, len(files)):
            # 0.8 as train dataset
            if i % 5 != 0:
                path_seq = []
                file = files[i]
                full_file_path = os.path.join(dataset_path, file)
                f = open(full_file_path, 'rt')
                for line in f:
                    line = line.strip('\n')
                    line = line.split(' ')

                    # read frame, angle, end point frame,
                    # the current position label, the next position label, the direction angle
                    line = [line[0], [float(line[1]), float(line[2])],
                            line[3],
                            int(line[4]),
                            int(line[5]), [float(line[6]), float(line[7])]
                            ]
                    path_seq.append(line)
                f.close()

                # if there are not enough input frames
                for j in range(1, len(path_seq) + 1):
                    if j - self.input_len < 0:
                        res_seq.append(path_seq[0: j])
                    else:
                        res_seq.append(path_seq[j - self.input_len: j])

        print(len(res_seq))
        self.imgs = res_seq

    def __len__(self):
        """
        return the length of the dataset
        :return:
        """
        return len(self.imgs)

    def __getitem__(self, index):
        """
        read the image sequence, angle sequence and label corresponding to the index in the dataset
        :param index: index of self.imgs
        :return: frame sequence, angle sequence,
                the current position label, the next position label, the direction angle
        """
        item = self.imgs[index]

        next_imgs = None
        next_angles = []

        # generate frame sequence and angle sequence
        for i in range(0, len(item)):
            img = item[i][0]
            img = Image.open(img)
            img = img.convert('RGB')
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


class OrderTestDataset(Dataset):
    def __init__(self, dataset_path, transform, input_len):
        """
        test dataset, form a sequence every five frames with an end point frame
        :param transform: torchvision.transforms
        :param input_len: input sequence length (not containing the end point)
        """
        self.transform = transform
        self.input_len = input_len

        res_seq = []
        files = os.listdir(dataset_path)
        files.sort()
        for i in range(0, len(files)):
            # 0.2 as test dataset
            if i % 5 == 0:
                path_seq = []
                file = files[i]
                full_file_path = os.path.join(dataset_path, file)
                f = open(full_file_path, 'rt')
                for line in f:
                    line = line.strip('\n')
                    line = line.split(' ')

                    # read frame, angle, end point frame,
                    # the current position label, the next position label, the direction angle
                    line = [line[0], [float(line[1]), float(line[2])],
                            line[3],
                            int(line[4]),
                            int(line[5]), [float(line[6]), float(line[7])]
                            ]
                    path_seq.append(line)
                f.close()

                # if there are not enough input frames
                for j in range(1, len(path_seq) + 1):
                    if j - self.input_len < 0:
                        res_seq.append(path_seq[0: j])
                    else:
                        res_seq.append(path_seq[j - self.input_len: j])

        print(len(res_seq))
        self.imgs = res_seq

    def __len__(self):
        """
        return the length of the dataset
        :return:
        """
        return len(self.imgs)

    def __getitem__(self, index):
        """
        read the image sequence, angle sequence and label corresponding to the index in the dataset
        :param index: index of self.imgs
        :return: frame sequence, angle sequence,
                the current position label, the next position label, the direction angle
        """
        item = self.imgs[index]

        next_imgs = None
        next_angles = []

        # generate frame sequence and angle sequence
        for i in range(0, len(item)):
            img = item[i][0]
            img = Image.open(img)
            img = img.convert('RGB')
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
        # output: the current position label, the next position label, the direction angle.
        return next_imgs, next_angles, label1, label2, label3
