from torch import nn

from network.NetworkABC import NetworkABC

from .u_net import U_Net


class UNet_Network(NetworkABC):
    def __init__(self, input_channels=3, num_classes=1) -> None:
        super().__init__(input_channels, num_classes)

    def get_network(self) -> nn.Module:
        """
        Genreic method to obtain netowrk
        """
        return U_Net(img_ch=self.input_channels, output_ch=self.num_classes)


__all__ = ["UNet_Network"]
