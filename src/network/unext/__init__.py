from torch import nn

from network.NetworkABC import NetworkABC

from .unext import UNext, UNextS


class UNeXt_Network(NetworkABC):
    def __init__(self, input_channels=3, num_classes=1) -> None:
        super().__init__(input_channels, num_classes)

    def get_network(self) -> nn.Module:
        return UNext(input_channels=self.input_channels, num_classes=self.num_classes)


class UNeXtS_Network(NetworkABC):
    def __init__(self, input_channels=3, num_classes=1) -> None:
        super().__init__(input_channels, num_classes)

    def get_network(self) -> nn.Module:
        return UNextS(input_channels=self.input_channels, num_classes=self.num_classes)


__all__ = ["UNeXt_Network", "UNeXtS_Network"]
