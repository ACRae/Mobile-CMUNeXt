from torch import nn

from network.NetworkABC import NetworkABC

from .ulite import ULite


class ULite_Network(NetworkABC):
    def __init__(self, input_channels=3, num_classes=1) -> None:
        super().__init__(input_channels, num_classes)

    def get_network(self) -> nn.Module:
        """
        Genreic method to obtain netowrk
        """
        return ULite(num_classes=self.num_classes, input_channel=self.input_channels)


__all__ = ["ULite_Network"]
