import os
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
folder_path = 'C:\\CYLab\\data\\GFP_VO\\'
path = 'C:\\CYLab\\data\\GFP_VO\\0015_A6_0001.tif'
image = cv2.imread(path)
# plt.imshow(image)
# plt.show()

#
#
# print(image)
#
# if os.path.exists(dir):
#     print("檔案路徑正確。")
# else:
#     print("檔案路徑不存在。")
# print(np.max(image[343][222])) #x 223,y 344: pixel:35


# image[y][x] [row_index][column_index]
# global index
# index = 0
# for i in range(1023):
#     for j in range(1023):
#         if(image[j][i][1]<45 and image[j][i][1]!= 0):
#             print(image[j][i][1])
#             index+=1
# print(index)
# print(round(37004/(1024*1024),3)*100) #37004
#3.5% below 45

# global num_images
# num_images=0
#
# for filename in os.listdir(folder_path):
#         if filename.endswith('.tif'):
#             num_images += 1
#             path = os.path.join(folder_path, filename)
# print('image num:', num_images) #num: 418

########################################
# Intersection judgement
# intersection = list_area
# intersection = [item[0] for item in list_area]
# for item in list_non_hollow:
#     if item in intersection:
#         intersection = [item]
#
# for item in list_net:
#     if item in intersection:
#         intersection = [item]
#
# print(f'Good Vessel Organoid: {intersection}')

import csv




# 示例用法
data = [
    ['Value1', 10, True],
    ['Value2', 20, False],
    ['Value3', 30, True]
]

# def write_to_csv(data, filename):
#     # 打开 CSV 文件并写入数据
#     with open(filename, mode='w', newline='') as file:
#         writer = csv.writer(file)
#
#         # 写入标题行
#         writer.writerow(['Column1', 'Column2', 'Column3'])
#
#         # 写入数据行
#         for row in data:
#             writer.writerow(row)
#
#     print(f'Data has been written to {filename} successfully.')
#
# f_path = 'C:\\CYLab\\data\\'
# filename = f_path + 'resultGoodVO.csv'
# write_to_csv(data, filename)

# Read csv
def read_csv_data(filename):
    # 读取 CSV 文件数据并返回一个列表
    data = []
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过标题行
        for row in reader:
            data.append(row)
    return data

f_path = 'C:\\CYLab\\data\\'
filename = f_path + 'all_res.csv'
# print(read_csv_data(filename))

import csv
######################################
# Intersection
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

    # print(f'Intersection.csv 文件已成功写入指定的路径：{csv_path}')

# 要进行交集操作的 CSV 文件路径
csv_file = f_path + 'all_res.csv'

find_intersection(csv_file)



########################################


########################################
# 自定義 HSV
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# # 創建一個空白影像
# image = np.zeros((400, 400, 3), dtype=np.uint8)
#
# # 圓心座標和半徑
# center = (200, 200)
# radius = 100
#
# # VO 皆介於 [h,s,v] [60,255,8]~[60,255,255]之間, 其中 V為45以下的有37004筆, 佔3.5%
# # 自訂 HSV 值
# h = 60  # 色調 (H: 0-179)
# s = 255  # 飽和度 (S: 0-255)
# v = 100  # 亮度 (V: 0-255)
#
# # 將 HSV 值轉換為 BGR 值
# hsv_color = np.array([[[h, s, v]]], dtype=np.uint8)
# bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)
# bgr_color = tuple(map(int, bgr_color[0, 0]))
#
# # 繪製圓形
# cv2.circle(image, center, radius, bgr_color, -1)

# 顯示圖像
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.show()


# Compare machine result with Intersection&Vote
import csv

def compare_results(tool_result_file, intersection_file, output_file):
    tool_results = read_csv_data(tool_result_file)
    intersection_results = read_csv_data(intersection_file)

    compared_results = []
    i=0
    for tool_row in tool_results:
        filename_tool = tool_row[0]
        result_tool = tool_row[1]
        # print(result_tool)
        matched = False
        for intersection_row in intersection_results:
            filename_intersection = intersection_row[0]
            result_intersection = intersection_row[1]
            # print(result_intersection)

            if filename_tool == filename_intersection:
                # i += 1
                # print(i) #293
                if result_tool == 'good' and result_intersection == 'good':
                    compared_results.append([filename_tool, 'good'])
                    # print(compared_results)
                elif result_tool == 'bad' and result_intersection == 'bad':
                    compared_results.append([filename_tool, 'bad'])
                    # print(compared_results)
                else:
                    compared_results.append([filename_tool, 'not matched'])
                    # print(compared_results)
                matched = True
                break

        if not matched:
            compared_results.append([filename_tool, 'not matched'])
    # print(len(compared_results))
    write_csv_data(output_file, compared_results)
    print(f"Comparison results have been written to {output_file} successfully.")

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
tool_csv = f_path + 'predictions.csv'

# Intersection.csv 文件路径
intersection_csv = f_path + 'Intersection.csv'

# 新的 CSV 文件路径
output_csv = f_path + 'Comparison_Result.csv'

compare_results(tool_csv, intersection_csv, output_csv)