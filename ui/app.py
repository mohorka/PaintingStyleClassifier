import os
from tkinter import (BOTTOM, LEFT, TOP, Button, Canvas, E, Frame, Label, Scrollbar, Tk, W,
                     filedialog, VERTICAL, X, HORIZONTAL, BOTH)

import cv2 as cv
import numpy as np
from core.config.config import RESIZE
from core.scripts.recognition import recognition
from copy import deepcopy
from PIL import Image, ImageTk
from keras.preprocessing.image import load_img, img_to_array


class App(Tk):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.title('Painting style classifier')
        self.attributes('-zoomed', True)
        self.interface_frame = Frame(self)
        self.interface_frame.pack(side=TOP, anchor=W)

        self.canvas_frame = Frame(self)
        self.canvas_frame.pack(side=TOP, anchor=W)

        self.label = Label(
            self.interface_frame,
            text='Choose a painting to get its style'
        )
        self.label.grid(row=0,column=0,padx=15,pady=8)

        self.download_painting = Button(
            self.interface_frame,
            text='Upload',
            width=15,
            command=self.upload,
            )
        self.download_painting.grid(row=1,column=0,padx=15,pady=8)

        self.recognize = Button(
            self.interface_frame,
            text='Go!',
            width=15,
            command=self.classify
        )
        self.recognize.grid(row=1,column=1,padx=15,pady=8)

        self.canvas = Canvas(self.canvas_frame, width=2000, height=900)
        self.canvas.pack(side=TOP)
       
        self.painting_label = Label(self.canvas, text='Painting: ')
        self.painting_label.pack(side=LEFT, anchor=E)
        self.canvas.create_window(100,30,window=self.painting_label)

        self.vgg16 = Label(self.canvas, text='VGG16: ')
        self.vgg16.pack(side=LEFT, anchor=E)
        self.canvas.create_window(500, 30, window=self.vgg16)

        self.vgg19 = Label(self.canvas, text='VGG19: ')
        self.vgg19.pack(side=LEFT, anchor=E)
        self.canvas.create_window(850, 30, window=self.vgg19)

        self.xception = Label(self.canvas, text='Xception: ')
        self.xception.pack(side=LEFT, anchor=E)
        self.canvas.create_window(1200, 30, window=self.xception)

        self.resnet50 = Label(self.canvas, text='ResNet50: ')
        self.resnet50.pack(side=LEFT, anchor=E)
        self.canvas.create_window(1600, 30, window=self.resnet50)

        self.inception_v3 = Label(self.canvas, text='Inception v3: ')
        self.inception_v3.pack(side=LEFT, anchor=E)
        self.canvas.create_window(100, 430, window=self.inception_v3)

        self.inception_resnet_v2 = Label(self.canvas, text='Inception Resnet v2: ')
        self.inception_resnet_v2.pack(side=LEFT, anchor=E)
        self.canvas.create_window(500, 430, window=self.inception_resnet_v2)

        self.result_label = Label(self.canvas, text='Style: ')
        self.canvas.create_window(850, 430, window=self.result_label)

        self.class_label = Label(self.canvas, text='')
        self.canvas.create_window(1200,430,window=self.class_label)

        self.painting = None
        self.result = []
    
    def upload(self):
        global painting
        filename = filedialog.askopenfilename()
        self.painting = load_img(filename,target_size=RESIZE)
        self.painting = img_to_array(self.painting)
        painting = deepcopy(self.painting)
        self.painting = self.painting.reshape((1, self.painting.shape[0], self.painting.shape[1], self.painting.shape[2]))
        print(self.painting.shape)
        print('next is size of img')
        #painting *= 255
        #painting = painting.astype(np.uint8)
        painting = Image.fromarray((painting * 1).astype(np.uint8)).convert('RGB')
        #painting = Image.fromarray(painting)
        painting.save('./uploaded/painting.jpg')
        painting = ImageTk.PhotoImage(painting)
        self.canvas.create_image(120,250, image=painting)
        

    def classify(self):
        if self.painting is None:
            raise Exception('Image is not uploaded')
        label = recognition([self.painting])

        #result_x, result_y = 450,300

        self.class_label.config(text=label)

        for res in os.listdir('./results/'):
            feature = Image.open('./results/'+res)
            feature = feature.resize((300,300))
            feature = ImageTk.PhotoImage(feature)
            self.result.append(feature)
            #self.canvas.create_image(result_x, result_y,image=feature)
        
        self.canvas.create_image(500,250,image=self.result[0])
        self.canvas.create_image(850,250,image=self.result[1])
        self.canvas.create_image(1200,250,image=self.result[2])
        self.canvas.create_image(1600,250,image=self.result[3])
        self.canvas.create_image(120,600,image=self.result[4])
        self.canvas.create_image(500,600,image=self.result[5])
     



        











        

        