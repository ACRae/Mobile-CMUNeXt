from torch import nn

from network.NetworkABC import NetworkABC

from .unetpp import UNetPlusPlus


class UNetpp_Network(NetworkABC):
    def __init__(self, input_channels=3, num_classes=1) -> None:
        super().__init__(input_channels, num_classes)

    def get_network(self) -> nn.Module:
        """
        Genreic method to obtain netowrk
        """
        return UNetPlusPlus(in_ch=self.input_channels, num_classes=self.num_classes)


__all__ = ["UNetpp_Network"]
