import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.lrelu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.in1 = nn.InstanceNorm2d(128)
        self.lrelu2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.in2 = nn.InstanceNorm2d(256)
        self.lrelu3 = nn.LeakyReLU(0.2, inplace=True)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.in3 = nn.InstanceNorm2d(512)
        self.lrelu4 = nn.LeakyReLU(0.2, inplace=True)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=4, stride=1, padding=1)
        self.in4 = nn.InstanceNorm2d(512)
        self.lrelu5 = nn.LeakyReLU(0.2, inplace=True)

        self.conv6 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)

        
    def forward(self, x):
        out = self.conv1(x)
        out = self.lrelu1(out)
        out = self.conv2(out)
        out = self.in1(out)
        out = self.lrelu2(out)
        out = self.conv3(out)
        out = self.in2(out)
        out = self.lrelu3(out)
        out = self.conv4(out)
        out = self.in3(out)
        out = self.lrelu4(out)
        out = self.conv5(out)
        out = self.in4(out)
        out = self.lrelu5(out)
        out = self.conv6(out)
        return out