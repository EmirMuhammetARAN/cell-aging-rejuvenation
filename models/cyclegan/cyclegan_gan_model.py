import torch
import torch.nn as nn

class CycleGANModel(nn.Module):
    def __init__(self, generator, discriminator, device):
        super(CycleGANModel, self).__init__()
        self.G_AB = generator()
        self.G_BA = generator()
        self.D_A = discriminator()
        self.D_B = discriminator()
        self.device = device

        self.criterion_GAN = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()

        self.optimizer_G = torch.optim.Adam(list(self.G_AB.parameters()) + list(self.G_BA.parameters()), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D_A = torch.optim.Adam(self.D_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D_B = torch.optim.Adam(self.D_B.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def set_input(self,input):
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
    
    def forward(self):
        self.fake_B = self.G_AB(self.real_A)
        self.rec_A = self.G_BA(self.fake_B)
        self.fake_A = self.G_BA(self.real_B)
        self.rec_B = self.G_AB(self.fake_A)
    
    def backward_G(self):
        pred_fake_B = self.D_B(self.fake_B)
        loss_G_AB = self.criterion_GAN(pred_fake_B, torch.ones_like(pred_fake_B))

        pred_fake_A = self.D_A(self.fake_A)
        loss_G_BA = self.criterion_GAN(pred_fake_A, torch.ones_like(pred_fake_A))

        loss_cycle_A = self.criterion_cycle(self.rec_A, self.real_A) * 10.0
        loss_cycle_B = self.criterion_cycle(self.rec_B, self.real_B) * 10.0

        loss_identity_A = self.criterion_identity(self.G_BA(self.real_A), self.real_A) * 5.0
        loss_identity_B = self.criterion_identity(self.G_AB(self.real_B), self.real_B) * 5.0

        loss_G = loss_G_AB + loss_G_BA + loss_cycle_A + loss_cycle_B + loss_identity_A + loss_identity_B

        return loss_G

    def backward_D(self,D,real,fake):
        pred_real = D(real)
        loss_D_real = self.criterion_GAN(pred_real, torch.ones_like(pred_real))

        pred_fake = D(fake.detach())
        loss_D_fake = self.criterion_GAN(pred_fake, torch.zeros_like(pred_fake))

        loss_D = (loss_D_real + loss_D_fake) * 0.5

        return loss_D
    
    def train_step(self,input,scaler):
        self.set_input(input)

        with torch.amp.autocast('cuda'):
            self.forward()

        for param in self.D_A.parameters():
            param.requires_grad_(False)
        for param in self.D_B.parameters():
            param.requires_grad_(False)

        self.optimizer_G.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            loss_G = self.backward_G()
        scaler.scale(loss_G).backward()
        scaler.step(self.optimizer_G)

        for param in self.D_A.parameters():
            param.requires_grad_(True)
        for param in self.D_B.parameters():
            param.requires_grad_(True)

        self.optimizer_D_A.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            loss_D_A = self.backward_D(self.D_A, self.real_A, self.fake_A)
        scaler.scale(loss_D_A).backward()
        scaler.step(self.optimizer_D_A)

        self.optimizer_D_B.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            loss_D_B = self.backward_D(self.D_B, self.real_B, self.fake_B)
        scaler.scale(loss_D_B).backward()
        scaler.step(self.optimizer_D_B)

        scaler.update()

        return loss_G, loss_D_A, loss_D_B