import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.rp1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0)
        self.in1 = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.rp2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=0)
        self.in2 = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.rp1(x)
        out = self.conv1(out)
        out = self.in1(out)
        out = self.relu(out)
        out = self.rp2(out)
        out = self.conv2(out)
        out = self.in2(out)
        out += residual
        return out
    

class GeneratorResNet(nn.Module):
    def __init__(self, num_residual_blocks=9):
        super(GeneratorResNet, self).__init__()
        self.rp1 = nn.ReflectionPad2d(3)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=0)
        self.in1 = nn.InstanceNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.in2 = nn.InstanceNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.in3 = nn.InstanceNorm2d(256)
        self.relu = nn.ReLU(inplace=True)

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(256, 256) for _ in range(num_residual_blocks)]
        )

        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.in4 = nn.InstanceNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.in5 = nn.InstanceNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.rp2 = nn.ReflectionPad2d(3)
        self.conv4 = nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=0)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.rp1(x)
        out = self.conv1(out)
        out = self.in1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.in2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.in3(out)
        out = self.relu(out)

        out = self.residual_blocks(out)

        out = self.deconv1(out)
        out = self.in4(out)
        out = self.relu(out)

        out = self.deconv2(out)
        out = self.in5(out)
        out = self.relu(out)

        out = self.rp2(out)
        out = self.conv4(out)
        out = self.tanh(out)

        return out