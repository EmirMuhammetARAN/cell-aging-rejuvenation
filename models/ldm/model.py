import torch.nn as nn
from diffusers import UNet2DModel, AutoencoderKL , DDPMScheduler



class CellLDM(nn.Module):
    def __init__(self):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        self.vae.requires_grad_(False)

        self.scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
        self.unet = UNet2DModel(
            sample_size=64,
            in_channels=4,
            out_channels=4,
            layers_per_block=2,
            block_out_channels=(128, 256, 512, 512),
            down_block_types=("DownBlock2D", "DownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
            class_embed_type="embedding",
            num_class_embeds=2
        )
    def forward(self, x, t, class_labels):
        return self.unet(x, t, class_labels)