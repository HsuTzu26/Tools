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

        self.browse_button = tk.Button(master, text='Upload Folder', command=self.browse)
        self.browse_button.place(relx=0.8, y=0)
        self.browse_button.config(font=('TkDefaultFont',14))

        self.good_button = tk.Button(master, text='Good', command=lambda: self.to_folder_csv('good'))
        self.good_button.place(x=180, rely=0.10, width=100, height=100)
        self.good_button.config(state='disabled', font=('TkDefaultFont',20), fg='#00FF00')

        self.bad_button = tk.Button(master, text='Bad', command=lambda: self.to_folder_csv('bad'))
        self.bad_button.place(x=510, rely=0.10, width=100, height=100)
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

    # def judge(self):
    #     if not self.name_entry.get():
    #         self.browse_button.config(state='disabled')
    #     else:
    #         self.browse_button.config(state='normal')

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
            # self.hist_button.config(state='normal')
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
            # self.hist_button.config(state='disabled')
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

root = tk.Tk()
my_gui = ImageClassificationTool(root)
root.mainloop()

