from torch import nn

from .vit_seg_modeling import (
    CONFIGS as CONFIGS_ViT_seg,  # noqa: N811
    VisionTransformer as ViT_seg,
)


class TransUnet(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super().__init__()
        config_vit = CONFIGS_ViT_seg["R50-ViT-B_16"]
        config_vit.n_classes = output_ch
        config_vit.n_skip = 3
        config_vit.patches.grid = (int(256 / 16), int(256 / 16))
        self.net = ViT_seg(config_vit, img_size=256, num_classes=output_ch).cuda()

    def forward(self, x):
        return self.net(x)
