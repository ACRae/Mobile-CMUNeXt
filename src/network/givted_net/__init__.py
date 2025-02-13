from torch import nn

from network.NetworkABC import NetworkABC

from .givetdnet import GIVTEDNet


class GIVTEDNet_Network(NetworkABC):
    def __init__(self, input_channels=3, num_classes=1):
        super().__init__(input_channels, num_classes)

    def get_network(self) -> nn.Module:
        """
        Generic method to obtain the GIVTEDNet network with specified parameters.
        """
        return GIVTEDNet(input_channels=self.input_channels, num_classes=self.num_classes)


__all__ = ["GIVTEDNet_Network"]
