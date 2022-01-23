import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class UNet(nn.Module):
    def __init__(self,filters,input_size=1,out_size=1):
        super().__init__()
        #filters=[16,32,64,128,256]
        #Down Block1
        self.conv1_1=nn.Conv2d(input_size,filters[0],kernel_size=3,padding=1)
        self.conv1_2=nn.Conv2d(filters[0],filters[0],kernel_size=3,padding=1)
        self.pool1=nn.MaxPool2d(2)
        #Block2
        self.conv2_1=nn.Conv2d(filters[0],filters[1],kernel_size=3,padding=1)
        self.conv2_2=nn.Conv2d(filters[1],filters[1],kernel_size=3,padding=1)
        self.pool2=nn.MaxPool2d(2)
        #Block 3 
        self.conv3_1=nn.Conv2d(filters[1],filters[2],kernel_size=3,padding=1)
        self.conv3_2=nn.Conv2d(filters[2],filters[2],kernel_size=3,padding=1)
        self.pool3=nn.MaxPool2d(2)     
        # Block 4
        self.conv4_1=nn.Conv2d(filters[2],filters[3],kernel_size=3,padding=1)
        self.conv4_2=nn.Conv2d(filters[3],filters[3],kernel_size=3,padding=1)
        self.pool4=nn.MaxPool2d(2)  
        
        # Bottleneck
        self.conv5_1=nn.Conv2d(filters[3],filters[4],kernel_size=3,padding=1)
        self.conv5_2=nn.Conv2d(filters[4],filters[4],kernel_size=3,padding=1)
        self.conv5_t=nn.ConvTranspose2d(filters[4],filters[3],kernel_size=2,stride=2)
        #Up Bloks [1]
        self.conv6_1=nn.Conv2d(filters[4],filters[3],kernel_size=3,padding=1)
        self.conv6_2=nn.Conv2d(filters[3],filters[3],kernel_size=3,padding=1)
        self.conv6_t=nn.ConvTranspose2d(filters[3],filters[2],kernel_size=2,stride=2)
        
        #Block [2]
        self.conv7_1=nn.Conv2d(filters[3],filters[2],kernel_size=3,padding=1)
        self.conv7_2=nn.Conv2d(filters[2],filters[2],kernel_size=3,padding=1)
        self.conv7_t=nn.ConvTranspose2d(filters[2],filters[1],kernel_size=2,stride=2)
        #Block [3]
        self.conv8_1=nn.Conv2d(filters[2],filters[1],kernel_size=3,padding=1)
        self.conv8_2=nn.Conv2d(filters[1],filters[1],kernel_size=3,padding=1)
        self.conv8_t=nn.ConvTranspose2d(filters[1],filters[0],kernel_size=2,stride=2)
        
        #Block [4]
        self.conv9_1=nn.Conv2d(filters[1],filters[0],kernel_size=3,padding=1)
        self.conv9_2=nn.Conv2d(filters[0],filters[0],kernel_size=3,padding=1)
        
        #Output 
        
        self.conv10=nn.Conv2d(filters[0],out_size,kernel_size=3,padding=1)
        
        
    def forward(self,x):
        #Down
        conv1=F.relu(self.conv1_1(x))
        conv1=F.relu(self.conv1_2(conv1))
        pool1=self.pool1(conv1)
        
        conv2=F.relu(self.conv2_1(pool1))
        conv2=F.relu(self.conv2_2(conv2))
        pool2=self.pool2(conv2)
        
        conv3=F.relu(self.conv3_1(pool2))
        conv3=F.relu(self.conv3_2(conv3))
        pool3=self.pool3(conv3)
        
        conv4=F.relu(self.conv4_1(pool3))
        conv4=F.relu(self.conv4_2(conv4))
        pool4=self.pool4(conv4)
        
        #Bottleneck
        
        conv5=F.relu(self.conv5_1(pool4))
        conv5=F.relu(self.conv5_2(conv5))
        
        #Up
        up6=torch.cat((self.conv5_t(conv5),conv4),dim=1)
        conv6=F.relu(self.conv6_1(up6))
        conv6=F.relu(self.conv6_2(conv6))
        
        up7=torch.cat((self.conv6_t(conv6),conv3),dim=1)
        conv7=F.relu(self.conv7_1(up7))
        conv7=F.relu(self.conv7_2(conv7))

        up8=torch.cat((self.conv7_t(conv7),conv2),dim=1)
        conv8=F.relu(self.conv8_1(up8))
        conv8=F.relu(self.conv8_2(conv8))
        
        
        up9=torch.cat((self.conv8_t(conv8),conv1),dim=1)
        conv9=F.relu(self.conv9_1(up9))
        conv9=F.relu(self.conv9_2(conv9))
        
        out=F.sigmoid(self.conv10(conv9))
        
        
        
        
        return out

    
        
    
