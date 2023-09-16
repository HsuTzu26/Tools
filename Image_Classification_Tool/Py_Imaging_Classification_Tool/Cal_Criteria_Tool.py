import os
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

folder_path = 'C:\\CYLab\\data\\GFP_VO\\'
path = 'C:\\CYLab\\data\\GFP_VO\\0015_A6_0001.tif'
image = cv2.imread(path)


# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# print(hsv.shape)
# global index
# index = 0
# for i in range(1023):
#     for j in range(1023):
#         if(hsv[j][i][2]<45 and hsv[j][i][2]!= 0):
#             print(hsv[j][i][2])
#             index+=1
# print(index) #37004


# print(np.max(hsv[343][222]))
# plt.imshow(hsv)
# plt.show()

def cal_mcst(path):
    threshold_area = 0.5
    threshold_value = 45

    # 讀入影像
    # path = 'data/exc 0018_C10_0001.tif'#false
    # path = 'data/exc 0017_A1_0002.tif'#false

    img = cv2.imread(path)
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # 只考慮像素值大於等於threshold_value的點
    threshold_mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)[1]

    # 計算影像中像素值大於等於threshold_value的點的數量
    threshold_pixels = cv2.countNonZero(threshold_mask)

    # 計算影像的總像素數量
    total_pixels = img.shape[0] * img.shape[1]

    # 設定蒙地卡羅演算法的迭代次數
    num_trials = total_pixels

    # 初始化特徵點數量的計數器
    num_features = 0

    # 進行蒙地卡羅演算法
    for i in range(num_trials):
        # 在影像中隨機取樣一個像素點
        x = np.random.randint(0, img.shape[1])
        y = np.random.randint(0, img.shape[0])

        # 如果該像素點之像素值大於等於threshold_value，則將特徵點數量的計數器加一
        if threshold_mask[y, x] == 255:
            num_features += 1

    # 計算特徵點數量的比例
    feature_ratio = num_features / num_trials

    # 計算特徵點數量的估計值
    estimated_features = int(feature_ratio * total_pixels)

    # judgement flag
    is_over = True

    # print(np.amax(threshold_mask, axis=1))
    # print(threshold_pixels)
    area_ratio = round((estimated_features / total_pixels), 3)
    print('螢光面積佔: ', area_ratio * 100, '%')
    # print('特徵點數量比例: ',format(feature_ratio, '.3f'))
    print('特徵點數量的估計值：', estimated_features)
    # print('像素值大於等於', threshold_value, '的點的數量：', threshold_pixels)

    if estimated_features / total_pixels > threshold_area:
        is_over = True
        print('螢光面積超過50%', is_over)
    else:
        is_over = False
        print('螢光面積未超過50%', is_over)

    # img_circle = cv2.circle(img, (512,512), 250, (0,0,225), -1)
    # print(250*250*3.14)

    plt.imshow(img)
    plt.show()


def cal_hsv(path):
    img = cv2.imread(path)
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # 將影像轉換為HSV色彩空間
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 定義綠色範圍的閾值
    lower_green = np.array([60, 50, 50])
    upper_green = np.array([180, 255, 255])

    # 提取綠色像素點的掩模
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # 計算影像中綠色像素點的數量
    green_pixels = cv2.countNonZero(mask)

    # 計算影像的總像素數量
    total_pixels = img.shape[0] * img.shape[1]

    # 設定蒙地卡羅演算法的迭代次數
    num_trials = 10000

    # 初始化特徵點數量的計數器
    num_features = 0

    # 進行蒙地卡羅演算法
    for i in range(num_trials):
        # 在影像中隨機取樣一個像素點
        x = np.random.randint(0, img.shape[1])
        y = np.random.randint(0, img.shape[0])

        # 如果該像素點是綠色的，則將特徵點數量的計數器加一
        if mask[y, x] == 255:
            num_features += 1

    # 計算特徵點數量的比例
    feature_ratio = num_features / num_trials

    # 計算特徵點數量的估計值
    estimated_features = int(feature_ratio * total_pixels)
    total = cv2.countNonZero(gray)

    print('特徵點數量的估計值：', estimated_features)
    # print(cv2.countNonZero(gray))
    area_ratio = round((estimated_features / total_pixels), 3)
    print('螢光面積佔: ', area_ratio * 100, '%')

    # if area_ratio > threshold_area:
    #   print('螢光面積超過50%')
    # else:
    #   print('螢光面積未超過50%')

    # plt.imshow(mask)
    # plt.show()
    # 顯示圖像
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(img)  # 原圖
    ax1.set_title('Original Image')
    ax2.imshow(mask)  # 處理後的圖
    ax2.set_title('Processed Image')
    plt.show()

def judge_hollow(path):
    img = cv2.imread(path)

    # 取得影像中心座標
    height, width, _ = img.shape
    center_x, center_y = width // 2, height // 2

    # 計算每個像素到中心點的距離
    distance = np.sqrt((np.arange(height)[:, np.newaxis] - center_y) ** 2 + (np.arange(width) - center_x) ** 2)

    # 找出距離中心點小於等於64的像素
    features = (distance <= 128).astype(np.uint8) * (img[:, :, 1] > 45).astype(np.uint8)

    # 計算特徵點數量
    num_features = np.sum(features)

    # 判斷是否為中空
    # is_hollow = num_features > 0
    is_hollow = True

    circle = 3.1415926 * 128 * 128
    threshold_hollow = 0.5
    hollow_ratio = round((num_features / circle),3)*100
    if hollow_ratio < threshold_hollow:
        # is_hollow = True
        print("Hollow")
    else:
        # is_hollow = False
        print('NonHollow')

        # print("特徵點數量:", num_features)
    print('Complete Rate: ', hollow_ratio)
    # print("圓面積", circle)
    # print("是否為中空:", not is_hollow)

    plt.imshow(img)
    plt.show()

def judge_network(path):
    # 讀取影像
    image = cv2.imread(path)

    # 影像預處理
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 轉成灰階
    _, binary_image = cv2.threshold(gray_image, 45, 255, cv2.THRESH_BINARY)  # 二值化
    blur_image = cv2.medianBlur(binary_image, 3)  # 中值濾波 減輕噪聲
    blur_image_twice = cv2.medianBlur(blur_image, 3)  # 平滑圖像

    # 進行形態學處理來去除噪點和小的斷點，可以使用開運算操作
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=2)
    # 使用Canny邊緣檢測算法來檢測影像中的邊緣
    canny = cv2.Canny(opening, 50, 150)
    # 尋找影像中的輪廓，可以使用cv2.findContours函數，設定輪廓檢測模式為cv2.RETR_TREE，輪廓近似方法為cv2.CHAIN_APPROX_SIMPLE
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 計算網絡分布的度量，可以計算影像中輪廓的數量、面積、周長等指標，這些指標可以用來評估網絡的分布情況
    num_contours = len(contours)
    total_area = sum([cv2.contourArea(cnt) for cnt in contours])
    total_length = sum([cv2.arcLength(cnt, True) for cnt in contours])
    # 根據網絡分布的度量結果，判斷網絡分布是否良好，可以根據先前的經驗設定一些閾值，例如，如果網絡分布的輪廓數量太少，面積太小，周長太短，就可以認為網絡分布不良好，否則可以認為網絡分布良好。

    print('輪廓數量: ', num_contours)
    print('輪廓面積: ', total_area)
    print('輪廓周長: ', total_length)

    is_WellNet = True

    if num_contours < 1300 and total_area < 18600 and total_length < 45000:
        print("Scarcity")
        # is_WellNet = False
    else:
        print("Abundant")
        # is_WellNet = True

    # print(binary_image[400])
    # print(blur_image[400])
    # print(blur_image_twice[400])
    # print(opening[400])

    # 顯示圖像
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(image)  # 原圖
    ax1.set_title('Original Image')
    ax2.imshow(canny)  # 處理後的圖
    ax2.set_title('Processed Image')
    plt.show()

def calculate_network_stats(folder_path):
    num_images = 0
    num_bad_networks = 0
    num_good_networks = 0
    total_contours = 0
    total_area = 0
    total_length = 0
    list_abundant=[]
    list_scarcity=[]

    c=0
    a=0
    l=0
    with tqdm(total=len(os.listdir(folder_path)), desc='Processing') as pbar:
        for filename in os.listdir(folder_path):
            if filename.endswith('.tif'):
                num_images += 1
                path = os.path.join(folder_path, filename)

                # 进行图像处理
                image = cv2.imread(path)
                # 影像預處理
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #轉成灰階
                _, binary_image = cv2.threshold(gray_image, 45, 255, cv2.THRESH_BINARY) #二值化
                blur_image = cv2.medianBlur(binary_image, 3) #中值濾波 減輕噪聲
                blur_image_twice = cv2.medianBlur(blur_image, 3) #平滑圖像

                # 進行形態學處理來去除噪點和小的斷點，可以使用開運算操作
                kernel = np.ones((3, 3), np.uint8)
                opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=2)
                # 使用Canny邊緣檢測算法來檢測影像中的邊緣
                canny = cv2.Canny(opening, 50, 150)
                # 尋找影像中的輪廓，可以使用cv2.findContours函數，設定輪廓檢測模式為cv2.RETR_TREE，輪廓近似方法為cv2.CHAIN_APPROX_SIMPLE
                contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                # 計算網絡分布的度量，可以計算影像中輪廓的數量、面積、周長等指標，這些指標可以用來評估網絡的分布情況
                num_contours = len(contours)
                total_area = sum([cv2.contourArea(cnt) for cnt in contours])
                total_length = sum([cv2.arcLength(cnt, True) for cnt in contours])

                # Flag judgement
                flag =1
                # 判断网络分布是否好
                # (好的數據+原本數據)取平均作修改
                # 原本數據colab(12) 1300 18600 45000 有點嚴苛 好的數據(149)# 759 27204 30248 太輕鬆
                # 使用 c,a,l重複更新後 1068 23104 34469 好的數據只有 4 太嚴格
                # 使用 c,a,l分別加上原數據和好的數據重複更新後 1197 32495 40522 好的數據只有 5 太嚴格

                # 以上結果合併後平均 1000 20000 37500
                # 結果為787.75 28405 31465 (164)

                # 平均下來 1000, 20000, 35000

                # 交集
                # good: 114, bad: 222
                # 眾數
                # good: 177, bad: 295?

                # 取接近交集的 1000, 25000, 35000 (111)
                # 905, 31356, 36036

                if num_contours < 1000 and total_area < 25000 and total_length < 35000:
                    flag=0
                    num_bad_networks += 1
                    list_scarcity.append([filename, flag])
                else:
                    flag=1
                    num_good_networks += 1
                    list_abundant.append([filename, flag])
                    c+=num_contours
                    a+=total_area
                    l+=total_length



                if num_images % 80 == 0 or num_images == len(os.listdir(folder_path)):
                    print(f"Processed {num_images} images.")

                # 在進度條中更新進度
                pbar.update(1)

                total_contours += num_contours

    c = c / num_good_networks
    a = a / num_good_networks
    l = l / num_good_networks

    # 计算网络分布比例
    avg_contours = (total_contours / num_images)
    avg_area = (total_area / num_images)
    avg_length = (total_length / num_images)
    good_network_ratio = round((num_good_networks / num_images),3)*100
    bad_network_ratio = round((num_bad_networks / num_images),3)*100
    print(c,a,l)
    print(num_good_networks)
    # print(num_images, total_contours, total_area, total_length)
    # print(f"Average number of contours: {avg_contours}")
    # print(f"Average area: {avg_area}")
    # print(f"Average length: {avg_length}")
    # print(f"Abundant network ratio: {good_network_ratio}%")
    # print(f"Scarcity network ratio: {bad_network_ratio}%")
    return list_abundant, list_scarcity
def calculate_hollow_stats(folder_path):
    num_images = 0
    num_non_hollow = 0
    num_hollow = 0
    stored = 0
    list_non_hollow=[]
    list_hollow = []

    with tqdm(total=len(os.listdir(folder_path)), desc='Processing') as pbar:
        for filename in os.listdir(folder_path):
            if filename.endswith('.tif'):
                num_images += 1
                path = os.path.join(folder_path, filename)

                img = cv2.imread(path)

                # 取得影像中心座標
                height, width, _ = img.shape
                center_x, center_y = width//2, height//2

                # 計算每個像素到中心點的距離
                distance = np.sqrt((np.arange(height)[:, np.newaxis] - center_y)**2 + (np.arange(width) - center_x)**2)

                # 找出距離中心點小於等於64的像素
                features = (distance <= 128).astype(np.uint8) * (img[:,:,1] > 45).astype(np.uint8)

                # 計算特徵點數量
                num_features = np.sum(features)

                # Flag judgement
                flag=1

                circle = 3.1415926 * 128* 128
                threshold_hollow = 0.5
                hollow_ratio = (num_features / circle)


                if hollow_ratio < threshold_hollow:
                  flag=1
                  num_hollow +=1
                  list_hollow.append([filename, hollow_ratio * 100])
                else:
                  flag=0
                  num_non_hollow+=1
                  list_non_hollow.append([filename, hollow_ratio*100])

                # print("特徵點數量:", num_features)
                # print('非中空率: ', hollow_ratio)
                stored += hollow_ratio*100
                # print("圓面積", circle)
                # print("是否為中空:", not is_hollow)
                if num_images % 80 == 0 or num_images == len(os.listdir(folder_path)):
                    print(f"Processed {num_images} images.")

                # 在進度條中更新進度
                pbar.update(1)

    avg_hollow_ratio = (stored / num_images) # 全部影像為中空的比率
    hollow_ratio = (num_hollow / num_images) * 100 # 為中空的比率
    non_hollow_ratio = (num_non_hollow / num_images) * 100 # 不為中空的比率
    print(f"Average hollow: {avg_hollow_ratio}%")
    print(f'Hollow ratio: {hollow_ratio}%')
    print(f'Non-hollow ratio: {non_hollow_ratio}%')
    return list_hollow, list_non_hollow

def calculate_area_stats(folder_path):
    num_images = 0
    num_good_networks = 0
    num_bad_networks = 0
    total_contours = 0
    total_area = 0
    total_length = 0
    stored = 0
    threshold_value = 45
    threshold_area_rate = 0.5

    list_OverFifty_area = []
    list_BellowFifty_area=[]

    with tqdm(total=len(os.listdir(folder_path)), desc='Processing') as pbar:
        for filename in os.listdir(folder_path):
            if filename.endswith('.tif'):
                num_images += 1
                path = os.path.join(folder_path, filename)

                img = cv2.imread(path)
                # gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

                # 將影像轉換為HSV色彩空間
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

                # 定義綠色範圍的閾值
                lower_green = np.array([60, 255, 45])
                upper_green = np.array([60, 255, 255])

                # 提取綠色像素點的掩模
                mask = cv2.inRange(hsv, lower_green, upper_green)

                # 計算影像的總像素數量
                total_pixels = img.shape[0] * img.shape[1]

                # 設定蒙地卡羅演算法的迭代次數
                num_trials = total_pixels

                # 初始化特徵點數量的計數器
                num_features = 0

                # 進行蒙地卡羅演算法
                for i in range(num_trials):
                    # 在影像中隨機取樣一個像素點
                    x = np.random.randint(0, img.shape[1])
                    y = np.random.randint(0, img.shape[0])

                    # 如果該像素點是綠色的，則將特徵點數量的計數器加一
                    if mask[y, x] == 255:
                        num_features += 1

                    # 計算特徵點數量的比例
                feature_ratio = num_features / num_trials

                # 計算特徵點數量的估計值
                estimated_features = int(feature_ratio * total_pixels)
                # 面積比例
                area_ratio = round((estimated_features / total_pixels), 3)

                # judgement flag
                is_over = True
                stored += area_ratio * 100

                flag = 1
                if estimated_features / total_pixels > threshold_area_rate:
                    is_over = True
                    # print(f'{filename}:', is_over)
                    # print('螢光面積超過50%', is_over)
                    flag = 1
                    num_good_networks += 1
                    # stored += area_ratio * 100

                    list_OverFifty_area.append([path, area_ratio * 100])

                else:
                    is_over = False
                    # print(f'{filename}:', is_over)
                    # print('螢光面積未超過50%', is_over)
                    flag = 0
                    num_bad_networks += 1

                    list_BellowFifty_area.append([path, area_ratio*100])

                if num_images %80 == 0 or num_images==len(os.listdir(folder_path)):
                    print(f"Processed {num_images} images.")

                # 在進度條中更新進度
                pbar.update(1)

        avg_area_ratio = (stored / num_images)
        overFifty_ratio = (num_good_networks / num_images)*100
        belowFifty_ratio = (num_bad_networks / num_images)*100

        print(f"Average area ratio: {avg_area_ratio}%")
        print(f'Over 50% area ratio: {overFifty_ratio}%')
        print(f'Below 50% area ratio: {belowFifty_ratio}%')
        # return list_OverFifty_area, list_BellowFifty_area




# cal_mcst(path) # Area 10.6%, estimate value: 111432
# cal_hsv(path) # Area 12.9%, estimate value: 135580 choose this
# judge_hollow(path) #  Hollow, Complete Rate 37.8%
# judge_network(path)

# calculate_area_stats(folder_path)
###########################################
# Average area ratio: 15.305502392344502%
# Over 50% area ratio: 0.23923444976076555%
# Below 50% area ratio: 99.76076555023924%
###########################################

# calculate_hollow_stats(folder_path)
# list_h, list_nh = calculate_hollow_stats(folder_path)
# print('Hollow:', list_h)
# print('Non-Hollow:',list_nh, '\n')
###########################################
# Average hollow: 41.62649470522987%
# Hollow ratio: 63.397129186602875%
# Non-hollow ratio: 36.60287081339713%
###########################################

calculate_network_stats(folder_path)
###########################################
# list_abundant, list_scarcity = calculate_network_stats(folder_path)
# Average number of contours: 503.57894736842104
# Average area: 1.1794258373205742
# Average length: 0.3621937045069973
# Abundant network ratio: 42.3%
# Scarcity network ratio: 57.699999999999996%
###########################################


import csv
###########################################
# Intersection judgement
# def write_to_csv(data, filename, result):
#     # 打开 CSV 文件并写入数据
#     with open(filename, mode='w', newline='') as file:
#         writer = csv.writer(file)
#
#         # 写入标题行
#         writer.writerow(['FileName', 'Result'])
#
#         # 写入数据行
#         for item in data:
#             writer.writerow([item, result])
#
#     print(f'Data has been written to {filename} successfully.')
#
# # Intersection for Good
# intersection = list_nh
# intersection = [item[0] for item in list_nh]
# for item in list_abundant:
#     if item in intersection:
#         intersection = [item]
# print(f'Good Vessel Organoid: {intersection}')
#
# f_path = 'C:\\CYLab\\data\\'
# filename = f_path + 'resultGoodVO.csv'
# write_to_csv(intersection, filename, 'Good')
#
# # Intersection for Bad
# intersection = list_h
# intersection = [item[0] for item in list_h]
# for item in list_scarcity:
#     if item in intersection:
#         intersection = [item]
# # print(f'Bad Vessel Organoid: {intersection}')
# f_path = 'C:\\CYLab\\data\\'
# filename = f_path + 'resultBadVO.csv'
# write_to_csv(intersection, filename, 'Bad')


#寫入同一csv
import csv

def write_results_to_csv(good_data, bad_data, filename):
    # 打开 CSV 文件并写入数据
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        # 写入标题行
        writer.writerow(['FileName', 'Result'])

        # 写入 Good 数据行
        for item in good_data:
            writer.writerow([item, 'Good'])

        # 写入 Bad 数据行
        for item in bad_data:
            writer.writerow([item, 'Bad'])

    print(f'Data has been written to {filename} successfully.')

# Intersection for Good
intersection_good = [item[0] for item in list_nh]
for item in list_abundant:
    if item in intersection_good:
        intersection_good = [item]
# print(f'Good Vessel Organoid: {intersection_good}')

# # Intersection for Bad
intersection_bad = list_h
intersection_bad = [item[0] for item in list_h]
for item in list_scarcity:
    if item in intersection_bad:
        intersection_bad = [item]
# print(f'Bad Vessel Organoid: {intersection_bad}')

f_path = 'C:\\CYLab\\data\\'
filename = f_path + 'Tool_Result.csv'
write_results_to_csv(intersection_good, intersection_bad, filename)

###################################
# all_res do intersection
import csv

def find_intersection(csv_file):
    # 读取 CSV 文件的数据
    data = read_csv_data(csv_file)

    # 获取交集结果
    intersection_result = calculate_intersection(data)

    # 将交集结果写入 Intersection.csv 文件
    intersection_csv_path = os.path.join(f_path, "Intersection.csv")
    write_intersection_csv(intersection_result, intersection_csv_path)

def read_csv_data(csv_file):
    # 读取 CSV 文件数据并返回一个列表
    data = []
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    return data

def calculate_intersection(data):
    # 获取交集结果
    intersection_result = []
    for row in data:
        filename = row[0]
        result = row[1:]
        if all(r == "good" for r in result):
            intersection_result.append([filename, "good"])
        elif all(r == "bad" for r in result):
            intersection_result.append([filename, "bad"])
    return intersection_result

def write_intersection_csv(data, csv_path):
    # 创建 Intersection.csv 文件并写入数据
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        # 写入标题行
        writer.writerow(['FileName', 'Intersection'])

        # 写入数据行
        writer.writerows(data)

    print(f'Intersection.csv 文件已成功写入指定的路径：{csv_path}')

# 要进行交集操作的 CSV 文件路径
csv_file = f_path + 'all_res.csv'

find_intersection(csv_file)

#####################################
# Vote
# Vote
import csv
from statistics import mode

def find_mode(csv_file):
    # 读取 CSV 文件的数据
    data = read_csv_data(csv_file)

    # 计算众数结果
    mode_result = calculate_mode(data)

    # 将众数结果写入 Vote.csv 文件
    vote_csv_path = os.path.join(f_path, "Vote.csv")
    write_vote_csv(mode_result, vote_csv_path)

def read_csv_data(csv_file):
    # 读取 CSV 文件数据并返回一个列表
    data = []
    with open(csv_file, mode='r', newline='') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过首行标题
        for row in reader:
            data.append(row)
    return data

def calculate_mode(data):
    # 计算众数结果
    mode_result = []
    for row in data:
        filename = row[0]
        result = row[1:]
        vote = mode(result)
        mode_result.append([filename, vote])
    return mode_result

def write_vote_csv(data, csv_path):
    # 创建 Vote.csv 文件并写入数据
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        # 写入标题行
        writer.writerow(['FileName', 'Vote'])

        # 写入数据行
        writer.writerows(data)

    print(f'Vote.csv 文件已成功写入指定路径：{csv_path}')

find_mode(csv_file)

##########################################
# Compare machine result with Intersection&Vote
import csv

def compare_results(tool_result_file, intersection_file, output_file):
    tool_results = read_csv_data(tool_result_file)
    intersection_results = read_csv_data(intersection_file)

    compared_results = []
    i=0
    for tool_row in tool_results:
        filename_tool = tool_row[0]
        result_tool = tool_row[1:]
        # print(result_tool)

        matched = False
        for intersection_row in intersection_results:
            filename_intersection = intersection_row[0]
            result_intersection = intersection_row[1:]
            # print(result_intersection)

            if filename_tool == filename_intersection:
                # i += 1
                # print(i) #293
                if result_tool == 'Good' and result_intersection == 'Good':
                    compared_results.append([filename_tool, 'Good'])
                    print(compared_results)
                elif result_tool == 'Bad' and result_intersection == 'Bad':
                    compared_results.append([filename_tool, 'Bad'])
                    # print(compared_results)
                else:
                    compared_results.append([filename_tool, 'none'])
                    # print(compared_results)
                matched = True
                break

        if not matched:
            compared_results.append([filename_tool, ''])
    # print(len(compared_results))
    write_csv_data(output_file, compared_results)
    # print(f"Comparison results have been written to {output_file} successfully.")

def read_csv_data(csv_file):
    data = []
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        for row in reader:
            data.append(row)
    return data

def write_csv_data(csv_file, data):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['FileName', 'ComparisonResult'])
        writer.writerows(data)
# Tool_Result.csv 文件路径
tool_csv = f_path + 'Tool_Result.csv'

# Intersection.csv 文件路径
intersection_csv = f_path + 'Intersection.csv'

# 新的 CSV 文件路径
output_csv = f_path + 'Comparison_Result.csv'

compare_results(tool_csv, intersection_csv, output_csv)

