from torch import nn

from network.NetworkABC import NetworkABC

from .unet_v2 import UNetV2


class UNetV2_Network(NetworkABC):
    def __init__(
        self,
        input_channels=3,
        num_classes=1,
        input_h=256,
        input_w=256,
    ) -> None:
        super().__init__(input_channels, num_classes, input_w, input_h)

    def get_network(self) -> nn.Module:
        return UNetV2(
            channel=self.input_channels,
            n_classes=self.num_classes,
            deep_supervision=False,
            pretrained_path=None,
        )


__all__ = ["UNetV2_Network"]
