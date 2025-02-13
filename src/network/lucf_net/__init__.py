from torch import nn

from network.NetworkABC import NetworkABC

from .lucf_net import LUCF_Net


class LUCFNet_Network(NetworkABC):
    def __init__(self, input_channels=3, num_classes=1) -> None:
        super().__init__(input_channels, num_classes)

    def get_network(self) -> nn.Module:
        """
        Genreic method to obtain netowrk
        """
        return LUCF_Net(in_chns=self.input_channels, class_num=self.num_classes)


__all__ = ["LUCFNet_Network"]
