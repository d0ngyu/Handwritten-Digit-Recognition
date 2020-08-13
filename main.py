# -*- coding:utf-8 -*-
import sys
from PyQt5.QtGui import QPixmap
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication,QMainWindow,QFileDialog
from functools import partial

import HDR

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

names=['0','1','2','3','4','5','6','7','8','9']

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(1,6,3,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(6,16,5),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.fc1=nn.Sequential(
            nn.Linear(16*5*5,120),
            nn.BatchNorm1d(120),
            nn.ReLU()
        )
        self.fc2=nn.Sequential(
            nn.Linear(120,84),
            nn.BatchNorm1d(84),
            nn.ReLU()
        )
        self.fc3=nn.Linear(84,10)

    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=x.view(x.size()[0],-1)
        x=self.fc1(x)
        x=self.fc2(x)
        x=self.fc3(x)
        return x

transform=transforms.Compose([
    transforms.ToTensor()
])

def open_image(ui):
    image_path, image_type = QFileDialog.getOpenFileName(None, "打开图片", "D:/", "*.png;;*.jpg;;All Files(*)")
    ui.lineEdit.setText(str(image_path))
    image = QPixmap(image_path)
    ui.label_2.setPixmap(image.scaled(28 * 4, 28 * 4))

def predict(model,device,add,transform):
    image=Image.open(add)
    image=transform(image)
    image=image.unsqueeze(0)
    image=image.to(device)

    model.eval()
    outputs=model(image)
    _,pred=torch.max(outputs,1)

    classIndex=pred[0]

    return names[classIndex]

def recognize(ui):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=torch.load('./models/lenet.pt')
    image_path=ui.lineEdit.text()

    result=predict(model,device,image_path,transform)

    ui.lineEdit_2.setText(result)

if __name__=='__main__':
    app=QApplication(sys.argv)
    MainWindow=QMainWindow()
    MainWindow.setWindowIcon(QIcon("./favicon.ico"))
    ui=HDR.Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()

    ui.pushButton.clicked.connect(partial(open_image,ui))
    ui.pushButton_2.clicked.connect(partial(recognize,ui))

    sys.exit(app.exec_())