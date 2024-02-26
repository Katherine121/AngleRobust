import PIL
import math
import random
import shutil
import numpy as np
import os
from PIL import ImageFile, Image
import imgaug.augmenters as iaa
ImageFile.LOAD_TRUNCATED_IMAGES = True


def deleteNone(path):
    """
    delete images int the dataset whose name contains "None"
    :param path: dataset path
    :return:
    """
    dir_path = os.listdir(path)
    dir_path.sort()
    print(len(dir_path))

    for dir in dir_path:
        full_dir_path = os.path.join(path, dir)

        file_path = os.listdir(full_dir_path)
        file_path.sort()

        pics_list = []
        for file in file_path:
            full_file_path = os.path.join(full_dir_path, file)

            # delete images int the dataset whose name contains "None"
            # if 'None' in full_file_path:
            #     os.remove(full_file_path)
            #     continue
            pics_list.append(full_file_path)
        print(len(pics_list))


def add_aug_order(dataset_path):
    # five kinds of disturbances
    image_augments = [iaa.Rain(), iaa.Snowflakes(), iaa.Fog(),
                      iaa.imgcorruptlike.Brightness(), iaa.Cutout()]

    dir_path = os.listdir(dataset_path)
    dir_path.sort()
    print(len(dir_path))

    noise = 0
    index = 97000
    for dir in dir_path:
        print(dir)
        # shuffle disturbances
        if noise % 5 == 0:
            random.shuffle(image_augments)

        full_dir_path = os.path.join(dataset_path, dir)
        new_full_dir_path = os.path.join(dataset_path, str(index))
        if os.path.exists(new_full_dir_path) is False:
            os.mkdir(new_full_dir_path)

        pic_list = os.listdir(full_dir_path)
        pic_list.sort()

        for pic in pic_list:
            full_pic_path = os.path.join(full_dir_path, pic)
            a = int(pic[0])
            a += 1
            new_pic = str(a) + pic[1:]
            new_full_pic_path = os.path.join(new_full_dir_path, new_pic)

            if os.path.exists(new_full_pic_path):
                continue

            pic = Image.open(full_pic_path)
            pic = pic.convert('RGB')

            # process image by imgaug
            pic = np.array(pic)
            image_augment = image_augments[noise % 5]
            pic = image_augment(image=pic)
            pic = Image.fromarray(pic)
            pic.save(new_full_pic_path)
        noise += 1
        index += 1


def get_pics(path):
    """
    get images and lat/lon labels from the dataset
    :param path: dataset path
    :return: image list, label list
    """
    pics_list = []
    labels = []

    dir_path = os.listdir(path)
    dir_path.sort()

    for dir in dir_path:
        full_dir_path = os.path.join(path, dir)

        file_path = os.listdir(full_dir_path)
        file_path.sort()
        print(len(file_path))

        for file in file_path:
            full_file_path = os.path.join(full_dir_path, file)

            pics_list.append(full_file_path)

            # latitude
            lat_index = file.find("lat")
            # altitude
            alt_index = file.find("alt")
            # longitude
            lon_index = file.find("lon")

            start = file[0: lat_index - 1]
            lat_pos = file[lat_index + 4: alt_index - 1]
            alt_pos = file[alt_index + 4: lon_index - 1]
            lon_pos = file[lon_index + 4: -4]

            labels.append(list(map(eval, [lat_pos, lon_pos])))

    return pics_list, labels


def euclidean_distance(pos1, pos2):
    """
    calculate Euclidean distance between two positions
    :param pos1: the first position
    :param pos2: the second position
    :return: the Euclidean distance
    """
    return math.sqrt(((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2))


def images_clustering(path, k, epoch):
    """
    cluster images in the dataset to different positions
    :param path: dataset path
    :param k: the number of clusters(positions)
    :param epoch: the number of cluster epochs
    :return: cluster center and cluster results
    """
    # get images and lat/lon labels from the dataset
    pics_list, labels = get_pics(path)
    print(len(pics_list))

    # randomly initialize k cluster centers
    centre = np.empty((k, 2))
    slot = len(labels) // k
    for i in range(0, k):
        centre[i][0] = labels[i * slot + slot // 2][0]
        centre[i][1] = labels[i * slot + slot // 2][1]

    # iterate epochs
    for iter in range(0, epoch):
        print(iter)

        # calculate the distance between the i-th pos and the j-th cluster center
        dis = np.empty((len(labels), k))
        for i in range(0, len(labels)):
            for j in range(0, k):
                dis[i][j] = euclidean_distance(labels[i], centre[j])

        # initialize the center results
        classify = []
        for i in range(0, k):
            classify.append([])

        # find the minimal distance and classify the pos to the correct cluster
        for i in range(0, len(labels)):
            List = dis[i].tolist()
            index = List.index(dis[i].min())
            classify[index].append(i)

        # initialize new k cluster centers based on the cluster results
        new_centre = np.empty((k, 2))
        for i in range(0, k):
            x_sum = 0
            y_sum = 0
            # avoid empty cluster
            if len(classify[i]) == 0:
                print("*************************")
                randindex = random.randint(0, len(labels) - 1)
                new_centre[i][0] = labels[randindex][0]
                new_centre[i][1] = labels[randindex][1]
                continue

            for j in range(0, len(classify[i])):
                x_sum += labels[classify[i][j]][0]
                y_sum += labels[classify[i][j]][1]

            new_centre[i][0] = x_sum / len(classify[i])
            new_centre[i][1] = y_sum / len(classify[i])

        # compare the old and the new cluster centers
        if (new_centre == centre).all():
            break
        else:
            centre = new_centre

    # record cluster results in files
    for i in range(0, k):
        if os.path.exists(str(k)) is False:
            os.mkdir(str(k))

        # record cluster centers
        with open(str(k) + "/cluster_centre.txt", "a") as file1:
            file1.write(str(centre[i][0]) + " " + str(centre[i][1]) + "\n")
        file1.close()

        # record image paths in a cluster
        with open(str(k) + "/cluster_pics.txt", "a") as file1:
            for j in range(0, len(classify[i])):
                ori_path = pics_list[classify[i][j]]
                file1.write(ori_path)
                if j == len(classify[i]) - 1:
                    file1.write("\n")
                else:
                    file1.write(" ")
        file1.close()

        # record image positions in a cluster
        with open(str(k) + "/cluster_labels.txt", "a") as file1:
            for j in range(0, len(classify[i])):
                file1.write(str(labels[classify[i][j]][0]) + " " + str(labels[classify[i][j]][1]))
                if j == len(classify[i]) - 1:
                    file1.write("\n")
                else:
                    file1.write(" ")
        file1.close()

    print('the number of iterations is: ', iter + 1)
    print('the cluster centers are: ', centre)

    return centre, classify


def copy_clusters(k):
    """
    copy cluster results(images) to corresponding k directories
    :param k: the number of clusters(positions)
    :return:
    """
    i = 0

    if os.path.exists(str(k) + "/all_class") is False:
        os.mkdir(str(k) + "/all_class")

    f = open(str(k) + "/cluster_pics.txt", 'rt')

    for line in f:
        line = line.replace('\n', '')
        line = line.split(' ')

        class_path = str(k) + "/all_class/" + str(i)
        os.mkdir(class_path)

        for j in range(0, len(line)):
            shutil.copy(line[j], class_path)

        i += 1
    f.close()


def get_cluster_labels(path1, k):
    """
    get the cluster(position) labels in routes
    :param k: the number of clusters(positions)
    :return:
    """
    path2 = str(k) + "/all_class"
    path3 = str(k) + "/cluster_labels"

    if os.path.exists(path3) is False:
        os.mkdir(path3)

    dir_path1 = os.listdir(path1)
    dir_path1.sort()

    for dir1 in dir_path1:
        file_path3 = os.path.join(path3, dir1 + ".txt")
        f1 = open(file_path3, "a")
        f1.close()

        full_dir_path1 = os.path.join(path1, dir1)

        file_path1 = os.listdir(full_dir_path1)
        file_path1.sort()

        print(dir1)
        print(len(file_path1))

        for file1 in file_path1:

            for i in range(0, k):
                full_dir_path2 = os.path.join(path2, str(i))
                full_file_path2 = os.path.join(full_dir_path2, file1)
                # if find the existance in i-th cluster directory,
                # the corresponding position label i is assigned
                if os.path.exists(full_file_path2):
                    with open(file_path3, "a") as f1:
                        f1.write(str(i) + "\n")
                    f1.close()
                    break


def replace_pics(k):
    """
    replace the images blank and non-readable with images in the same cluster
    :param k: the number of clusters(positions)
    :return:
    """
    res = []

    path1 = str(k) + "/all_class"
    a = 0

    for i in range(0, k):
        path2 = os.path.join(path1, str(i))

        files_for_one_class = os.listdir(path2)
        files_for_one_class.sort()

        for j in range(0, len(files_for_one_class)):
            full_file_path = os.path.join(path2, files_for_one_class[j])

            try:
                pic = Image.open(full_file_path)
                pic = pic.convert('RGB')
            except(OSError, PIL.UnidentifiedImageError):
                a += 1
                # replace the images blank and non-readable with images in the same cluster
                # +5, -5 make sure in the same train dataset or in the same test dataset
                if j - 5 >= 0:
                    alter_full_file_path = os.path.join(path2, files_for_one_class[j - 5])
                    alter_pic = Image.open(alter_full_file_path)
                    # save images to origin path
                    alter_pic.save(full_file_path)
                elif j + 5 < len(files_for_one_class):
                    alter_full_file_path = os.path.join(path2, files_for_one_class[j + 5])
                    alter_pic = Image.open(alter_full_file_path)
                    # save images to origin path
                    alter_pic.save(full_file_path)

            res.append((full_file_path, i))
    print(a)
    print(len(res))


def copy_replaced_pics(path1, k):
    """
    replace the images blank and non-readable in the origin dataset with images in the corresponding cluster
    :param k: the number of clusters(positions)
    :return:
    """
    res = []

    path2 = str(k) + "/all_class"

    dir1 = os.listdir(path1)
    dir1.sort()

    a = 0

    for dir in dir1:
        full_dir_path = os.path.join(path1, dir)

        files = os.listdir(full_dir_path)
        files.sort()

        for file in files:
            full_file_path = os.path.join(full_dir_path, file)

            try:
                pic = Image.open(full_file_path)
                pic = pic.convert('RGB')
            except(OSError):
                a += 1
                # replace the images blank and non-readable in the origin dataset with images in the corresponding cluster.
                for i in range(0, k):
                    full_dir_path2 = os.path.join(path2, str(i))
                    full_file_path2 = os.path.join(full_dir_path2, file)

                    if os.path.exists(full_file_path2):
                        alter_pic = Image.open(full_file_path2)
                        alter_pic.save(full_file_path)
                        break
            res.append(full_file_path)
    print(a)
    print(len(res))


def prepare_dataset(path1, k):
    """
    prepare for dataset
    :param k: the number of clusters(positions)
    :return:
    """
    path2 = str(k) + "/all_class"
    res = []

    # get all images and positions
    dir_path1 = os.listdir(path1)
    dir_path1.sort()

    for i in range(0, len(dir_path1)):
        dir1 = dir_path1[i]
        full_dir_path1 = os.path.join(path1, dir1)

        file_path1 = os.listdir(full_dir_path1)
        file_path1.sort()

        path = []

        for file1 in file_path1:
            full_file_path1 = os.path.join(full_dir_path1, file1)

            # latitude
            lat_index = file1.find("lat")
            # altitude
            alt_index = file1.find("alt")
            # longitude
            lon_index = file1.find("lon")

            start = file1[0: lat_index - 1]
            lat_pos = file1[lat_index + 4: alt_index - 1]
            alt_pos = file1[alt_index + 4: lon_index - 1]
            lon_pos = file1[lon_index + 4: -4]

            for j in range(0, k):
                full_dir_path2 = os.path.join(path2, str(j))
                full_file_path2 = os.path.join(full_dir_path2, file1)
                if os.path.exists(full_file_path2):
                    # frame, position label, coordinates
                    path.append((full_file_path1, j, eval(lat_pos), eval(lon_pos)))
                    break

        res.append(path)

    # calculate frame, angle, end point frame,
    # the current position label, the next position label, the direction angle
    # for every frame
    res_delta = []
    for path in res:
        path_delta = []

        for i in range(0, len(path)):
            flag = False
            for j in range(i + 1, len(path)):
                # if find the next position (if it is not the end point)
                if path[j][1] != path[i][1]:
                    # calculate the direction angle to arrive at the next position
                    lat_delta = (path[j][2] - path[i][2]) * 111000
                    lon_delta = (path[j][3] - path[i][3]) * 111000 * math.cos(path[i][2] / 180 * math.pi)
                    sum = math.sqrt(lat_delta * lat_delta + lon_delta * lon_delta)
                    sin = lat_delta / sum
                    cos = lon_delta / sum

                    # part of final labels: the next position label, the direction angle to arrive at the next position
                    stone_part = (path[j][1], sin, cos)

                    # calculate the direction angle at the next moment
                    lat_delta = (path[i + 1][2] - path[i][2]) * 111000
                    lon_delta = (path[i + 1][3] - path[i][3]) * 111000 * math.cos(path[i][2] / 180 * math.pi)
                    sum = math.sqrt(lat_delta * lat_delta + lon_delta * lon_delta)
                    sin = lat_delta / sum
                    cos = lon_delta / sum

                    # part of model input: the direction angle at the next moment
                    next_part = (sin, cos)

                    # exit
                    flag = True
                    break

            # if it is the end point, discard from dataset
            if flag is False:
                break

            # if it is not the end point, add it to dataset

            # frame, angle, end point frame,
            # the current position, the next position, the direction angle
            path_delta.append((path[i][0], next_part, path[-1][0], path[i][1], stone_part))

        res_delta.append(path_delta)

    # record in files
    if os.path.exists("datasets") is False:
        os.mkdir("datasets")
    for path in res_delta:
        for pic in path:
            name, next_part, dest_part, label_part, stone_part = pic[0], pic[1], pic[2], pic[3], pic[4]
            new_txt_name = name[len(path1) + 1: len(path1) + 1 + 5]

            name = "processOrder/" + name
            dest_part = "processOrder/" + dest_part
            with open("datasets/" + new_txt_name + ".txt", "a") as file1:
                file1.write(name + " " + str(next_part[0]) + " " + str(next_part[1]) + " " +
                            dest_part + " " +
                            str(label_part) + " " +
                            str(stone_part[0]) + " " + str(stone_part[1]) + " " + str(stone_part[2]) + "\n"
                            )
            file1.close()


if __name__ == "__main__":
    # # 1. delete images int the dataset whose name contains "None"
    # deleteNone(path="order")
    # # 2. delete excess images at the beginning and end manually
    # # 3. delete the extra images 75986:15, 83832:3, 97128: all
    # # we can find the dataset in this step in the USB drive
    # # add weather noise into the dataset
    # add_aug_order()
    # 4. cluster images in the dataset to different positions
    images_clustering(path="order", k=100, epoch=400)
    # 5. copy cluster results(images) to corresponding k directories
    copy_clusters(k=100)
    # 6. get the position labels in routes
    get_cluster_labels(path1="order", k=100)

    # # 7. replace the images blank and non-readable with images in the same cluster
    # # 8. replace the images blank and non-readable in the origin dataset with images in the corresponding cluster
    # replace_pics(k=100)
    # copy_replaced_pics(path1="order", k=100)

    # 9. calculate frame, angle, end point frame,
    #    the current position, the next position, the direction angle
    #    for every frame
    prepare_dataset(path1="order", k=100)
