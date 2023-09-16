import tkinter as tk
from tkinter.constants import * # anchor

# 建立主視窗和 Frame (把元件變成群組的容器)
window = tk.Tk()
# 設定視窗標題、大小和背景顏色
window.title('Imaging Classification Tool')
window.geometry('800x600')
window.configure(background='white')

# create new font
from tkinter.font import Font
new_font = Font(family='Times New Roman', size=12)


from tkinter import filedialog
loadFile_en = tk.Entry(width=40)
loadFile_en.place(x=70 ,y=0)
def loadFile():
    if loadFile_en.get() is None:
        file_path = filedialog.askopenfilename(
            filetypes = (("VO images","*.tif"),("all files","*.*")))
        loadFile_en.insert(0,file_path)
    else:
        file_path = filedialog.askopenfilename(
            filetypes = (("VO images","*.tif"),("all files","*.*")))
        loadFile_en.delete(0,'end')
        loadFile_en.insert(0,file_path)

# Upload Folder Entry
loadFile_btn = tk.Button(window, font=new_font ,text='Upload Folder', command=loadFile)
loadFile_btn.pack(side='top', anchor=NE)


window.mainloop()
