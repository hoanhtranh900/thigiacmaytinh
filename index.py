#Thêm thư viện tkinter
from tkinter import *
from tkinter import filedialog

#Tạo một cửa sổ mới
window = Tk()

#Thêm tiêu đề cho cửa sổ
window.title('Detect Mask')

#Đặt kích thước của cửa sổ
window.geometry('500x500')


btn1 = Button(window, text='Select Image', width=20, height=2)


#Thêm nút chọn ảnh từ webcam
btn2 = Button(window, text='Webcam', width=20, height=2)

btn1.pack(side=TOP, pady=10)
btn2.pack(side=TOP, pady=10)


#select image from local use askopenfilename
def selectImage():
    root = Tk()
    root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
    print (root.filename)
    

def checkMask():
    print("check mask")
    


# add click for btn1
btn1.config(command=selectImage)


#Lặp vô tận để hiển thị cửa sổ
window.mainloop()