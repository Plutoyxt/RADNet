import json

import os, cv2

from PIL import Image

import numpy as np

import json

import os, cv2

from scipy.stats import pearsonr

import random

train_json = '/media/ExtDisk/yxt/ssdd_coco-20221019/ssdd_coco/train1/train.json'
test_json = '/media/ExtDisk/yxt/ssdd_coco-20221019/ssdd_coco/test1/test.json'
val_json = '/media/ExtDisk/yxt/ssdd_coco-20221019/ssdd_coco/val1/val.json'
test_path = '/media/ExtDisk/yxt/ssdd_coco-20221019/ssdd_coco/test1/test_image/'
val_path = '/media/ExtDisk/yxt/ssdd_coco-20221019/ssdd_coco/val1/val_image/'
train_path = '/media/ExtDisk/yxt/ssdd_coco-20221019/ssdd_coco/train1//train_image/'

#hrsid
train_json1 = '/media/ExtDisk/yxt/HRSID/annotations/train2017.json'
test_json1 = '/media/ExtDisk/yxt/HRSID/annotations/test2017.json'
train_path1 = '/media/ExtDisk/yxt/HRSID/train_image/'
test_path1 = '/media/ExtDisk/yxt/HRSID/test_image/'

def getFileList(dir, Filelist, ext=None):
    """

    获取文件夹及其子文件夹中文件列表

    输入 dir：文件夹根目录

    输入 ext: 扩展名

    返回： 文件路径列表

    """

    newDir = dir

    if os.path.isfile(dir):

        if ext is None:

            Filelist.append(dir)

        else:

            if ext in dir[-3:]:
                Filelist.append(dir)



    elif os.path.isdir(dir):

        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)

            getFileList(newDir, Filelist, ext)

    return Filelist


def embedding_distance(feature_1, feature_2):
    feature_1 = np.array(feature_1)

    feature_2 = np.array(feature_2)

    dist = np.linalg.norm(feature_1 - feature_2)

    return dist


def sigmoid_bianhua(x, p1):
    p = (np.log((1 / 0.99) - 1) + 6) / (p1)

    x = p * (x)

    x = x - 6

    return 1 / (1 + np.exp(x))  # sigmoid函数


def guiyihua(p, max, min):
    d = p - min

    s = max - min

    d = d / s

    return (1 - d)


def listcaculate(u1, d1):
    f = []

    for i in range(0, len(u1)):
        f.append(int(u1[i]) - int(d1[i]))

    return f


def visualization_bbox1(num_image, json_path, img_path):
    with open(json_path) as annos:
        annotation_json = json.load(annos)
        image_name = annotation_json['images'][num_image - 1]['file_name']  # 读取图片名
        id = annotation_json['images'][num_image - 1]['id']  # 读取图片id
        image_path = os.path.join(img_path, str(image_name).zfill(5))  # 拼接图像路径
        print(str(image_path))
        image = cv2.imread(image_path, 1)  # 保持原始格式的方式读取图像
        num_bbox = 0  # 统计一幅图片中bbox的数量
        num_small = 0
        img = Image.open(image_path)
        img_array = np.array(img)  # 把图像转成数组格式img = np.asarray(image)

        dscore = []
        cscore = []
        last_score = []
        shape = img_array.shape

        for i in range(shape[0] * shape[1]):
            dscore.append(0)

        for i in range(shape[0] * shape[1]):
            cscore.append(0)

        for i in range(shape[0] * shape[1]):
            last_score.append(0)
        for i in range(len(annotation_json['annotations'][::])):
            if annotation_json['annotations'][i - 1]['image_id'] == id:
                num_bbox = num_bbox + 1
                point = []
                distance = []
                x_Y = []
                corr = []
                cscore1 = []
                x, y, w, h = annotation_json['annotations'][i - 1]['bbox']  # 读取边框
                if w * h <= 1024:
                    num_small = num_small + 1

                x1 = x + w / 2
                y1 = y + h / 2  # (x1,y1) 为中心点 (x2,y2)为右下角点 (x,y)为右上角点
                x2 = x + w
                y2 = y + h

                if x1 >= shape[1]:
                    x1 = shape[1] - 1

                if y1 >= shape[0]:
                    y1 = shape[0] - 1

                if x2 >= shape[1]:
                    x2 = shape[1] - 1

                if y2 >= shape[0]:
                    y2 = shape[0] - 1

                # 修改中心点颜色
                print("x1,y1:", max(w / 2, h / 2), x1, y1, shape[0], shape[1])
                img_array[int(y1), int(x1)] = (255, 255, 255)
                center = (int(y1), int(x1))
                # 得到集合Y

                for i in range(int(y), int(y2)):
                    for j in range(int(x), int(x2)):
                        c = embedding_distance(list(img_array[int(y1), int(x1)]), list(img_array[int(i), int(j)]))
                        if c < max(w / 2, h / 2):
                            point.append([i, j])

                            # img_array[i,j]=(255,0,0)

                # 计算每个像素点与中心点的距离

                for i in range(0, shape[1]):
                    for j in range(0, shape[0]):
                        # i是纵坐标，j是横坐标
                        dis = embedding_distance([y1, x1], [j, i])
                        # print(dis)
                        distance.append([dis, [j, i]])

                # 计算距离分数
                R = max(w / 2, h / 2)
                r1 = 1.5 * R
                for i in range(0, len(distance)):
                    if dscore[i] < sigmoid_bianhua(distance[i][0], r1):
                        dscore[i] = sigmoid_bianhua(distance[i][0], r1)
                if len(point)>=10:
                    point1 = random.sample(point, 10)
                else:
                    point1 = random.sample(point, len(point))

                print(point1)
                for i in range(0, shape[1]):
                    for j in range(0, shape[0]):
                        d = 0
                        x_point = []
                        # print(i,j)
                        while d != len(point1):
                            x_point.append(listcaculate((img_array[(point1[d])[0], (point1[d])[1]]), img_array[j, i]))
                            d = d + 1
                        # print(x_point)
                        x_Y.append(x_point)
                for i in range(0, len(x_Y)):
                    zjz = np.array(x_Y[i])

                    p_rg = pearsonr(zjz[:, 0], zjz[:, 1])
                    p_rb = pearsonr(zjz[:, 0], zjz[:, 2])
                    p_gb = pearsonr(zjz[:, 1], zjz[:, 2])

                    p_rg = abs(p_rg[0])
                    p_rb = abs(p_rb[0])
                    p_gb = abs(p_gb[0])

                    cor = [[1, p_rg, p_rb], [p_rg, 1, p_gb], [p_rb, p_gb, 1]]
                    corr.append(np.array(cor))
                print(corr[0])

                for i in range(0, len(x_Y)):
                    a = np.dot(np.array(x_Y[i]), corr[i])
                    a = np.dot(a, np.transpose(a))
                    shape1 = a.shape
                    z = 0
                    z1 = 0

                    for j in range(0, shape1[0]):
                        for k in range(0, shape1[1]):
                            if j == k:
                                z = z + a[j, k]
                                z1 = z1 + 1

                    z = (1 / z1) * z
                    z = np.sqrt(z)
                    cscore1.append(z)

                print("cscore1",max(cscore1), min(cscore1))

                print("daxiao",len(cscore1), shape[0] * shape[1], len(distance))

                for i in range(0, len(cscore1)):

                    #if dscore[i] >= 0.94 and dscore[i] <= 1:
                    if dscore[i] >= 0 and dscore[i] <= 1:
                        if (guiyihua(cscore1[i], max(cscore1), min(cscore1))) > cscore[i]:
                            cscore[i] = guiyihua(cscore1[i], max(cscore1), min(cscore1))

                        # print(i)

                print("d",max(dscore), min(dscore))

        alpha = 0.7

        if num_small / num_bbox != 0:
            b = 0
        else:
            b = 1 #no have small targets
        a = 0.94
        theta = a
        print("a:", theta,b)

        for i in range(0, len(distance)):
            if b==1 :
              if dscore[i] <= 0.94:
                last_score[i] = 0
              elif dscore[i] >= 0.97 and dscore[i] <=1:
              #if dscore[i] >= 0.97 and dscore[i] <= 1:
                  last_score[i] = alpha * dscore[i] + (1 - alpha) * cscore[i]
                  #print("score:",dscore[i],last_score[i])
              else:
              #elif dscore[i] >=0 and dscore[i] < 0.97:
                  last_score[i] = (1 - alpha) * dscore[i] + alpha * (1 - cscore[i])
                  print("score:", dscore[i], last_score[i])
            else:
               if dscore[i] <= 0.94:
                   last_score[i] = 0
               else:
                    last_score[i]=1
        for i in range(0, len(last_score)):
            if last_score[i] < 0.64:#(theta/2):
                last_score[i] = 0
        for i in range(0, len(distance)):
            if last_score[i] == 0:
                img_array[(distance[i][1])[0], (distance[i][1])[1]] = (0, 0, 0)
        img2 = Image.fromarray(np.uint8(img_array))
        #img2.save("/media/ExtDisk/yxt/ssdd_coco-20221019/ssdd_coco/val1/ONLYCOLOR/" + str(image_name).zfill(5), 'png')
        cv2.imwrite("/media/ExtDisk/yxt/HRSID/ONLYCOLOR/" + str(image_name).zfill(5),img_array)

if __name__ == "__main__":

    d = 0

    #org_img_folder = '/media/ExtDisk/yxt/ssdd_coco-20221019/ssdd_coco/val1/val_image/'
    org_img_folder = '/media/ExtDisk/yxt/HRSID/test_image/'
    imglist = getFileList(org_img_folder, [], 'jpg')

    for imgpath in imglist:
        d = d + 1
        visualization_bbox1(d, test_json1, test_path1)
        # print(d)
        # 距离加色差