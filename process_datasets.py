import math
import os


def prepare_dataset(center_lat=23.4, center_lon=120.3,
                    path1="../../../mnt/nfs/wyx/taiwan",
                    num_nodes=100,
                    radius=10000,
                    square_nodes=30):
    """
    prepare for dataset.
    :return:
    """

    # 生成900个矩形中心点
    square_centers = []

    lat_diff = 20000 / square_nodes / 111000
    lon_diff = 20000 / square_nodes

    for i in range(0, square_nodes):
        lat = round(center_lat - 0.1 + i * lat_diff, 6)
        for j in range(0, square_nodes):
            lon = round(center_lon - 0.1 + j * lon_diff / 111000 / math.cos(lat / 180 * math.pi), 6)
            square_centers.append([lat, lon])

    # 生成圆形中心点
    circle_centers = []
    for center in square_centers:
        lat_diff = (center[0] - center_lat) * 111000
        lon_diff = (center[1] - center_lon) * 111000 * math.cos(center_lat / 180 * math.pi)
        diff = math.sqrt(lat_diff * lat_diff + lon_diff * lon_diff)
        if diff <= radius:
            circle_centers.append(center)
    print(len(circle_centers))

    # get all images and positions
    res = []
    dict = {}
    for i in range(0, num_nodes, 5):
        for j in range(i + 1, num_nodes, 5):
            print(str(i) + "-" + str(j))
            for k in range(0, 20):
                full_dir = os.path.join(path1, "path" + str(i) + "," + str(j) + "," + str(k))

                files = os.listdir(full_dir)
                files.sort(key=lambda x: int(x[0: x.find(',')]))

                path = []

                for file1 in files:
                    full_file_path1 = os.path.join(full_dir, file1)

                    corrdinates = file1[:-4]
                    corrdinates = corrdinates.split(',')
                    lat_pos = float(corrdinates[1])
                    lon_pos = float(corrdinates[2])

                    min_dis = math.inf
                    min_k = -1
                    for index in range(0, len(circle_centers)):
                        center = circle_centers[index]
                        lat_diff = (lat_pos - center[0]) * 111000
                        lon_diff = (lon_pos - center[1]) * 111000 * math.cos(center[0] / 180 * math.pi)
                        diff = math.sqrt(lat_diff * lat_diff + lon_diff * lon_diff)
                        if diff <= min_dis:
                            min_dis = diff
                            min_k = index
                    path.append((full_file_path1, min_k, lat_pos, lon_pos))
                    if dict.get(min_k) is None:
                        dict[min_k] = 1
                    else:
                        dict[min_k] += 1

                res.append(path)

    print(dict)
    index = max(dict.keys(), key=(lambda x: dict[x]))
    # 输出最大value对应的key
    print(index)
    # 输出最大value
    print(dict[index])

    # calculate frame, angle, end point frame,
    # the current position label, the next position label, the direction angle
    # for every frame
    res_delta = []
    for path in res:
        path_delta = []

        for i in range(0, len(path)):
            if i < len(path) - 1:
                # calculate the direction angle at the next moment
                lat_delta = (path[i + 1][2] - path[i][2]) * 111000
                lon_delta = (path[i + 1][3] - path[i][3]) * 111000 * math.cos(path[i][2] / 180 * math.pi)
                sum = math.sqrt(lat_delta * lat_delta + lon_delta * lon_delta)
                sin = lat_delta / sum
                cos = lon_delta / sum
                # part of model input: the direction angle at the next moment
                img_ang = (sin, cos)

                target = i
                for j in range(i + 1, len(path)):
                    if path[j][1] != path[i][1]:
                        target = j
                        break

                if target != i:
                    # calculate the direction angle from the next position
                    lat_delta = (path[target][2] - path[i][2]) * 111000
                    lon_delta = (path[target][3] - path[i][3]) * 111000 * math.cos(path[i][2] / 180 * math.pi)
                    sum = math.sqrt(lat_delta * lat_delta + lon_delta * lon_delta)
                    target_sin = lat_delta / sum
                    target_cos = lon_delta / sum
                    # part of model output: the direction angle from the next position
                    target_part = (path[target][1], target_sin, target_cos)
                else:
                    target_part = (path[i][1], sin, cos)

                # if it is not the end point, add it to dataset
                # frame, angle, end point frame,
                # the current position, the next position, the direction angle from the next position
                path_delta.append((path[i][0], img_ang, path[-1][0], path[i][1], target_part))

        res_delta.append(path_delta)

    # record in files
    if os.path.exists("../../../mnt/nfs/wyx/datasets") is False:
        os.mkdir("../../../mnt/nfs/wyx/datasets")

    num = 0
    for i in range(0, num_nodes, 5):
        for j in range(i + 1, num_nodes, 5):
            print(str(i) + "-" + str(j))
            for k in range(0, 20):
                path = res_delta[num]

                with open("../../../mnt/nfs/wyx/datasets/path" +
                          str(i) + "," + str(j) + "," + str(k) + ".txt", "a") as file1:
                    for pic in path:
                        name, img_ang, dest_part, label_part, target_part = pic[0], pic[1], pic[2], pic[3], pic[4]

                        file1.write(name + " " + str(img_ang[0]) + " " + str(img_ang[1]) + " " +
                                    dest_part + " " +
                                    str(label_part) + " " +
                                    str(target_part[0]) + " " + str(target_part[1]) + " " + str(target_part[2]) + "\n"
                                    )
                file1.close()

                num += 1


if __name__ == "__main__":
    # 9. calculate frame, angle, end point frame,
    #    the current position, the next position, the direction angle
    #    for every frame
    prepare_dataset()
