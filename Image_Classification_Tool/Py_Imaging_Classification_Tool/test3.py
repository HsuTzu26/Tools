import random
import tkinter as tk
from pprint import pprint
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import csv
import os
import shutil
from PIL import ImageEnhance, ImageOps

# imaging processing
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot

class ImageClassificationTool:
    def __init__(self, master):
        self.master = master
        master.title('Image Classification Tool')
        master.geometry('800x600')

        # initial variables
        self.fileCount = 0
        self.index = 0
        self.dir = ''
        self.csvPath = ''
        self.tmp = 0
        self.user = ''
        self.files = []
        self.files_list = []

        # create GUI elements
        self.label = tk.Label(master, text='Please enter user name:')
        self.label.pack()
        self.label.config(font=('TkDefaultFont',12))

        self.name_entry = tk.Entry(master)
        self.name_entry.pack()

        # Show the estimate value
        self.Area_txt = tk.Text(self.cal_Area())
        Area_txt.pack()


        self.browse_button = tk.Button(master, text='Upload Folder', command=self.browse)
        self.browse_button.place(relx=0.8, y=0)
        self.browse_button.config(font=('TkDefaultFont',14))

        self.good_button = tk.Button(master, text='Good', command=lambda: self.to_folder_csv('good'))
        self.good_button.place(x=180, rely=0.15, width=100, height=100)
        self.good_button.config(state='disabled', font=('TkDefaultFont',20), fg='#00FF00')

        self.bad_button = tk.Button(master, text='Bad', command=lambda: self.to_folder_csv('bad'))
        self.bad_button.place(x=510, rely=0.15, width=100, height=100)
        self.bad_button.config(state='disabled', font=('TkDefaultFont',20), fg='#CC0000')


        # Sharpen & Histogram Equalization
        self.sharpen_button = tk.Button(master, text='Sharpen', command=self.sharpen_image)
        self.sharpen_button.pack()
        self.sharpen_button.config(state='disabled', font=('TkDefaultFont',14))

        # self.hist_button = tk.Button(master, text='Histogram Equalization', command=self.hist_eq_image)
        # self.hist_button.pack()
        # self.hist_button.config(state='disabled', font=('TkDefaultFont',16))

        # self.quit_button = tk.Button(master, text='Quit', command=master.quit)
        # self.quit_button.pack()

        self.canvas1 = tk.Canvas(master, width=300, height=300)
        self.canvas1.pack(side=tk.LEFT, padx=50, pady=50)

        # self.frame = tk.Frame(master, padx=300)
        # self.frame.pack(side='left')

        self.canvas2 = tk.Canvas(master, width=300, height=300)
        self.canvas2.pack(side=tk.LEFT, padx=50, pady=50)

        # create empty csv file
        self.create_csv()

    def judge(self):
        if not self.name_entry.get():
            self.browse_button.config(state='disabled')
        else:
            self.browse_button.config(state='normal')

    # browse and open folder
    def browse(self):
        if not self.name_entry.get():
            messagebox.showerror('Error', 'Please enter user name.')
            return

        self.dir = filedialog.askdirectory()
        if self.dir:
            self.index = -1
            self.files = []
            self.shuffle()
            self.change_image()
            self.good_button.config(state='normal')
            self.bad_button.config(state='normal')
            self.sharpen_button.config(state='normal')
            self.hist_button.config(state='normal')
            self.create_csv()
            self.name_entry.delete(0, 1000)


    # create empty csv file
    def create_csv(self):
        if not self.name_entry.get():
            flag=1
        else:
            self.user = self.name_entry.get() + '.csv'
            self.csvPath = os.path.join(self.dir, self.user)
            with open(self.csvPath, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['FileName', 'Result'])

    # shuffle
    def shuffle(self):
        #randomly select next image
        self.files = set([f for f in os.listdir(self.dir) if f.endswith('.tif')])
        self.files_list = list(self.files)
        random.shuffle(self.files_list)
        pprint(self.files_list)

    # change image in canvas
    def change_image(self):
        # self.files = set([f for f in os.listdir(self.dir) if f.endswith('.tif')])
        self.fileCount = len(self.files)
        if self.index == self.fileCount - 1:
            self.good_button.config(state='disabled')
            self.bad_button.config(state='disabled')
            self.sharpen_button.config(state='disabled')
            self.hist_button.config(state='disabled')
        else:
            self.index += 1

        image_path = os.path.join(self.dir, self.files_list[self.index])
        image = Image.open(image_path)
        image1 = image.resize((300, 300), Image.LANCZOS) # LANCZOS
        self.image_tk = ImageTk.PhotoImage(image1)
        self.canvas1.create_image(0, 0, anchor='nw', image=self.image_tk)
        image2 = image.resize((300, 300), Image.LANCZOS) #ANTIALIAS
        self.image_tk2 = ImageTk.PhotoImage(image2)
        self.canvas2.create_image(0, 0, anchor='nw', image=self.image_tk2)


    # copy image to good or bad folder and write to csv
    def to_folder_csv(self, result):
        tmp = self.index

        source_path = os.path.join(self.dir, f'{result}')
        if not os.path.exists(source_path):
            os.mkdir(source_path)
        destination_path = os.path.join(source_path, self.files_list[tmp])
        source_path = os.path.join(self.dir, self.files_list[tmp])
        shutil.copy2(source_path, destination_path)
        self.change_image()
        # write to csv
        with open(self.csvPath, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.files_list[tmp], result])

    def sharpen_image(self):
        image_path = os.path.join(self.dir, self.files_list[self.index])
        image = Image.open(image_path)
        sharpened = ImageEnhance.Sharpness(image).enhance(6)
        sharpened = sharpened.resize((300, 300), Image.LANCZOS)  # ANTIALIAS
        self.image_tk2 = ImageTk.PhotoImage(sharpened)
        self.canvas2.create_image(0, 0, anchor='nw', image=self.image_tk2)

        # create Sharpen folder if it doesn't exist
        sharpen_path = os.path.join(self.dir, 'Sharpen')
        if not os.path.exists(sharpen_path):
            os.mkdir(sharpen_path)
        # sharpen image
        sharpened = ImageEnhance.Sharpness(image).enhance(6)

        # save sharpened image
        sharpened_path = os.path.join(sharpen_path, self.files_list[self.index])
        sharpened.save(sharpened_path)


    # def hist_eq_image(self):
    #     image_path = os.path.join(self.dir, self.files_list[self.index])
    #     image = Image.open(image_path)
    #     hist_eq = ImageOps.equalize(image)
    #     hist_eq = hist_eq.resize((300, 300), Image.LANCZOS)  # ANTIALIAS
    #     self.image_tk2 = ImageTk.PhotoImage(hist_eq)
    #     self.canvas2.create_image(0, 0, anchor='nw', image=self.image_tk2)
    #
    #     # create HistoEq folder if it doesn't exist
    #     histo_eq_path = os.path.join(self.dir, 'HistoEq')
    #     if not os.path.exists(histo_eq_path):
    #         os.mkdir(histo_eq_path)
    #     # equalize histogram
    #     histo_eq_image = ImageOps.equalize(image)
    #
    #     # save histogram equalized image
    #     histo_eq_path = os.path.join(histo_eq_path, self.files_list[self.index])
    #     histo_eq_image.save(histo_eq_path)

    def cal_Area(self):

        import matplotlib.pyplot as plt

        threshold_area = 0.5
        threshold_value = 45

        # 讀入影像
        image_path = os.path.join(self.dir, self.files_list[self.index])
        img = cv2.imread(image_path)
        gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

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

        # plt.imshow(img)
        # plt.show()
        return is_over, area_ratio

    def judge_hollow(self):

        # 讀取影像
        image_path = os.path.join(self.dir, self.files_list[self.index])

        img = cv2.imread(image_path)

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
        hollow_ratio = num_features / circle
        if hollow_ratio < threshold_hollow:
            is_hollow = True
            print("為中空", is_hollow)
        else:
            is_hollow = False
            print('非中空', is_hollow)

            # print("特徵點數量:", num_features)
        print('非中空率: ', hollow_ratio)
        # print("圓面積", circle)
        # print("是否為中空:", not is_hollow)

        # plt.imshow(img)
        # plt.show()
        return is_hollow

    def estimate_network(self):
        # 讀取影像
        image_path = os.path.join(self.dir, self.files_list[self.index])
        image = cv2.imread(image_path)

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

        # print('輪廓數量: ', num_contours)
        # print('輪廓面積: ', total_area)
        # print('輪廓周長: ', total_length)

        is_WellNet = True

        if num_contours < 1300 and total_area < 18600 and total_length < 45000:
            print("Bad Network")
            is_WellNet = False
        else:
            print("Good Network")
            is_WellNet = True
        return is_WellNet

    def judgement(self):
        if self.cal_Area()==True and self.judge_hollow()==False and self.estimate_ntwork()==True:
            print("Good Vessel Organoid")
        else:
            print("Bad Vessel Organoid")




root = tk.Tk()
my_gui = ImageClassificationTool(root)
root.mainloop()
