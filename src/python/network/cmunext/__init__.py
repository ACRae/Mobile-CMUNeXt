from torch import nn

from network.NetworkABC import NetworkABC

from .cmunext import CMUNeXt
from .cmunext_add import CMUNeXt_Add


class CMUNeXt_Network(NetworkABC):
    def __init__(self, input_channels=3, num_classes=1) -> None:
        super().__init__(input_channels, num_classes)

    def get_network(self) -> nn.Module:
        return CMUNeXt(
            input_channel=self.input_channels,
            num_classes=self.num_classes,
            dims=[16, 32, 128, 160, 256],
            depths=[1, 1, 1, 3, 1],
            kernels=[3, 3, 7, 7, 7],
            act=nn.GELU
        )


class CMUNeXt_LeakyRELU_Network(NetworkABC):
    def __init__(self, input_channels=3, num_classes=1) -> None:
        super().__init__(input_channels, num_classes)

    def get_network(self) -> nn.Module:
        return CMUNeXt(
            input_channel=self.input_channels,
            num_classes=self.num_classes,
            dims=[16, 32, 128, 160, 256],
            depths=[1, 1, 1, 3, 1],
            kernels=[3, 3, 7, 7, 7],
            act=nn.LeakyReLU
        )

class CMUNeXt_RELU_Network(NetworkABC):
    def __init__(self, input_channels=3, num_classes=1) -> None:
        super().__init__(input_channels, num_classes)

    def get_network(self) -> nn.Module:
        return CMUNeXt(
            input_channel=self.input_channels,
            num_classes=self.num_classes,
            dims=[16, 32, 128, 160, 256],
            depths=[1, 1, 1, 3, 1],
            kernels=[3, 3, 7, 7, 7],
            act=nn.ReLU
        )

class CMUNeXt_Hardswish_Network(NetworkABC):
    def __init__(self, input_channels=3, num_classes=1) -> None:
        super().__init__(input_channels, num_classes)

    def get_network(self) -> nn.Module:
        return CMUNeXt(
            input_channel=self.input_channels,
            num_classes=self.num_classes,
            dims=[16, 32, 128, 160, 256],
            depths=[1, 1, 1, 3, 1],
            kernels=[3, 3, 7, 7, 7],
            act=nn.Hardswish
        )



class CMUNeXtS_Network(NetworkABC):
    def __init__(self, input_channels=3, num_classes=1) -> None:
        super().__init__(input_channels, num_classes)

    def get_network(self) -> nn.Module:
        return CMUNeXt(
            input_channel=self.input_channels,
            num_classes=self.num_classes,
            dims=[8, 16, 32, 64, 128],
            depths=[1, 1, 1, 1, 1],
            kernels=[3, 3, 7, 7, 9],
        )

class CMUNeXtS_LeakyRELU_Network(NetworkABC):
    def __init__(self, input_channels=3, num_classes=1) -> None:
        super().__init__(input_channels, num_classes)

    def get_network(self) -> nn.Module:
        return CMUNeXt(
            input_channel=self.input_channels,
            num_classes=self.num_classes,
            dims=[8, 16, 32, 64, 128],
            depths=[1, 1, 1, 1, 1],
            kernels=[3, 3, 7, 7, 9],
            act=nn.LeakyReLU
        )

class CMUNeXtS_RELU_Network(NetworkABC):
    def __init__(self, input_channels=3, num_classes=1) -> None:
        super().__init__(input_channels, num_classes)

    def get_network(self) -> nn.Module:
        return CMUNeXt(
            input_channel=self.input_channels,
            num_classes=self.num_classes,
            dims=[8, 16, 32, 64, 128],
            depths=[1, 1, 1, 1, 1],
            kernels=[3, 3, 7, 7, 9],
            act=nn.ReLU
        )

class CMUNeXtS_Hardswish_Network(NetworkABC):
    def __init__(self, input_channels=3, num_classes=1) -> None:
        super().__init__(input_channels, num_classes)

    def get_network(self) -> nn.Module:
        return CMUNeXt(
            input_channel=self.input_channels,
            num_classes=self.num_classes,
            dims=[8, 16, 32, 64, 128],
            depths=[1, 1, 1, 1, 1],
            kernels=[3, 3, 7, 7, 9],
            act=nn.Hardswish
        )



class CMUNeXtS_Kernel_3_Network(NetworkABC):
    def __init__(self, input_channels=3, num_classes=1) -> None:
        super().__init__(input_channels, num_classes)

    def get_network(self) -> nn.Module:
        return CMUNeXt(
            input_channel=self.input_channels,
            num_classes=self.num_classes,
            dims=[8, 16, 32, 64, 128],
            depths=[1, 1, 1, 1, 1],
            kernels=[3, 3, 3, 3, 3],
        )




class CMUNeXtL_Network(NetworkABC):
    def __init__(self, input_channels=3, num_classes=1) -> None:
        super().__init__(input_channels, num_classes)

    def get_network(self) -> nn.Module:
        return CMUNeXt(
            input_channel=self.input_channels,
            num_classes=self.num_classes,
            dims=[32, 64, 128, 256, 512],
            depths=[1, 1, 1, 6, 3],
            kernels=[3, 3, 7, 7, 7],
            act=nn.GELU
        )

class CMUNeXtL_LeakyRELU_Network(NetworkABC):
    def __init__(self, input_channels=3, num_classes=1) -> None:
        super().__init__(input_channels, num_classes)

    def get_network(self) -> nn.Module:
        return CMUNeXt(
            input_channel=self.input_channels,
            num_classes=self.num_classes,
            dims=[32, 64, 128, 256, 512],
            depths=[1, 1, 1, 6, 3],
            kernels=[3, 3, 7, 7, 7],
            act=nn.LeakyReLU
        )

class CMUNeXtL_RELU_Network(NetworkABC):
    def __init__(self, input_channels=3, num_classes=1) -> None:
        super().__init__(input_channels, num_classes)

    def get_network(self) -> nn.Module:
        return CMUNeXt(
            input_channel=self.input_channels,
            num_classes=self.num_classes,
            dims=[32, 64, 128, 256, 512],
            depths=[1, 1, 1, 6, 3],
            kernels=[3, 3, 7, 7, 7],
            act=nn.ReLU
        )

class CMUNeXtL_Hardswish_Network(NetworkABC):
    def __init__(self, input_channels=3, num_classes=1) -> None:
        super().__init__(input_channels, num_classes)

    def get_network(self) -> nn.Module:
        return CMUNeXt(
            input_channel=self.input_channels,
            num_classes=self.num_classes,
            dims=[32, 64, 128, 256, 512],
            depths=[1, 1, 1, 6, 3],
            kernels=[3, 3, 7, 7, 7],
            act=nn.Hardswish
        )


class CMUNeXtAddS_Network(NetworkABC):
    def __init__(self, input_channels=3, num_classes=1) -> None:
        super().__init__(input_channels, num_classes)

    def get_network(self) -> nn.Module:
        return CMUNeXt_Add(
            input_channel=self.input_channels,
            num_classes=self.num_classes,
            dims=[8, 16, 32, 64, 128],
            depths=[1, 1, 1, 1, 1],
            kernels=[3, 3, 7, 7, 9],
        )


class CMUNeXtXXS_Network(NetworkABC):
    def __init__(self, input_channels=3, num_classes=1) -> None:
        super().__init__(input_channels, num_classes)

    def get_network(self) -> nn.Module:
        return CMUNeXt(
            input_channel=self.input_channels,
            num_classes=self.num_classes,
            dims=[8, 10, 12, 16, 24],
            depths=[3, 1, 1, 1, 2],
            kernels=[3, 3, 7, 7, 9],
        )
