import random
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

import cv2
from PIL import Image, ImageTk
import os
import shutil
from PIL import ImageEnhance

class ImageClassificationTool:
    def __init__(self, master):
        self.master = master
        master.title('Image Classification Tool')
        master.geometry('800x600')

        # initial variables
        self.fileCount = 0
        self.index = 0
        self.dir = ''
        self.files = []
        self.files_list = []

        # create GUI elements
        self.label = tk.Label(master, text='Please enter user name:')
        self.label.pack()
        self.label.config(font=('TkDefaultFont', 12))

        self.name_entry = tk.Entry(master)
        self.name_entry.pack()

        self.browse_button = tk.Button(master, text='Upload Folder', command=self.browse)
        self.browse_button.place(relx=0.8, y=0)
        self.browse_button.config(font=('TkDefaultFont', 14))

        self.good_button = tk.Button(master, text='Good', command=lambda: self.to_folder('good'))
        self.good_button.place(x=180, rely=0.10, width=100, height=100)
        self.good_button.config(state='disabled', font=('TkDefaultFont', 20), fg='#00FF00')

        self.bad_button = tk.Button(master, text='Bad', command=lambda: self.to_folder('bad'))
        self.bad_button.place(x=510, rely=0.10, width=100, height=100)
        self.bad_button.config(state='disabled', font=('TkDefaultFont', 20), fg='#CC0000')

        self.canvas1 = tk.Canvas(master, width=300, height=300)
        self.canvas1.pack(side=tk.LEFT, padx=50, pady=50)

        self.canvas2 = tk.Canvas(master, width=300, height=300)
        self.canvas2.pack(side=tk.LEFT, padx=50, pady=50)

    def browse(self):
        # if not self.name_entry.get():
        #     messagebox.showerror('Error', 'Please enter user name.')
        #     return

        self.dir = filedialog.askdirectory()
        if self.dir:
            self.index = -1
            self.files = []
            self.shuffle()
            self.change_image()
            self.good_button.config(state='normal')
            self.bad_button.config(state='normal')
            self.name_entry.delete(0, tk.END)

    def shuffle(self):
        self.files = [f for f in os.listdir(self.dir) if f.endswith('.png') or f.endswith('.jpg')]
        self.files_list = random.sample(self.files, len(self.files))

    def change_image(self):
        if self.index == len(self.files_list) - 1:
            self.good_button.config(state='disabled')
            self.bad_button.config(state='disabled')
        else:
            self.index += 1

        # image_path = os.path.join(self.dir, self.files_list[self.index])
        # image = Image.open(image_path).convert('L')  # Convert to grayscale
        # image1 = image.resize((300, 300), Image.LANCZOS)
        # self.image_tk = ImageTk.PhotoImage(image1)
        # self.canvas1.create_image(0, 0, anchor='nw', image=self.image_tk)
        # image2 = image.resize((300, 300), Image.LANCZOS)
        # self.image_tk2 = ImageTk.PhotoImage(image2)
        # self.canvas2.create_image(0, 0, anchor='nw', image=self.image_tk2)
        image_path = os.path.join(self.dir, self.files_list[self.index])
        image = cv2.imread(image_path)
        image = cv2.resize(image, (300, 300))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.image_tk = self.convert_to_tk_image(image)
        self.canvas1.create_image(0, 0, anchor='nw', image=self.image_tk)

        self.image_tk2 = self.convert_to_tk_image(image)
        self.canvas2.create_image(0, 0, anchor='nw', image=self.image_tk2)

    def convert_to_tk_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image_tk = ImageTk.PhotoImage(image)
        return image_tk

    def to_folder(self, result):
        tmp = self.index

        source_path = os.path.join(self.dir, result)
        if not os.path.exists(source_path):
            os.mkdir(source_path)
        destination_path = os.path.join(source_path, self.files_list[tmp])
        source_path = os.path.join(self.dir, self.files_list[tmp])
        shutil.copy2(source_path, destination_path)
        self.change_image()

root = tk.Tk()
my_gui = ImageClassificationTool(root)
root.mainloop()
