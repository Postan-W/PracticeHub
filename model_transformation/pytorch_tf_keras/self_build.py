import torch
import torch.nn.functional as functional

class MnistClassificationDynamicInput(torch.nn.Module):
    def __init__(self,h,w):
        super(MnistClassificationDynamicInput,self).__init__()
        h1 = int((h-5+1)/2)
        w1 = int((w-5+1)/2)
        h2 = int((h1-4)/2)
        w2 = int((w1-4)/2)
        self.convolution1 = torch.nn.Conv2d(1,10,kernel_size=5)#输入单通道，输出10通道，卷积核大小(5,5)
        self.convolution2 = torch.nn.Conv2d(10,20,kernel_size=5)#输入10通道，输出20通道，卷积核大小(5,5)
        self.convolution2_drop = torch.nn.Dropout2d()
        self.k = 20*h2*w2
        print("全连接层的输入大小为:",self.k)
        self.full_connection1 = torch.nn.Linear(self.k,60)
        self.full_connection2 = torch.nn.Linear(60,10)


    def forward(self,x):
        x = functional.relu(functional.max_pool2d(self.convolution1(x),2))#此时尺寸为(10,12,12)
        x = functional.relu(functional.max_pool2d(self.convolution2_drop(self.convolution2(x)),2))#此时尺寸为(20,4,4)，所以共320个神经元
        x = x.view(-1,self.k)#展平操作
        x = functional.relu(self.full_connection1(x))
        x = functional.dropout(x,training=self.training)#training=True,训练的时候才dropout
        x = self.full_connection2(x)
        return functional.log_softmax(x)