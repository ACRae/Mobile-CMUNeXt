from .tikzeng import *


# define new block
def block_2ConvPool(
    name, botton, top, s_filer=256, n_filer=64, offset="(1,0,0)", size=(32, 32, 3.5), opacity=0.5
):
    return [
        to_ConvConvRelu(
            name=f"ccr_{name}",
            s_filer=str(s_filer),
            n_filer=(n_filer, n_filer),
            offset=offset,
            to=f"({botton}-east)",
            width=(size[2], size[2]),
            height=size[0],
            depth=size[1],
        ),
        to_Pool(
            name=f"{top}",
            offset="(0,0,0)",
            to=f"(ccr_{name}-east)",
            width=1,
            height=size[0] - int(size[0] / 4),
            depth=size[1] - int(size[0] / 4),
            opacity=opacity,
        ),
        to_connection(f"{botton}", f"ccr_{name}"),
    ]


def block_Unconv(
    name, botton, top, s_filer=256, n_filer=64, offset="(1,0,0)", size=(32, 32, 3.5), opacity=0.5
):
    return [
        to_UnPool(
            name=f"unpool_{name}",
            offset=offset,
            to=f"({botton}-east)",
            width=1,
            height=size[0],
            depth=size[1],
            opacity=opacity,
        ),
        to_ConvRes(
            name=f"ccr_res_{name}",
            offset="(0,0,0)",
            to=f"(unpool_{name}-east)",
            s_filer=str(s_filer),
            n_filer=str(n_filer),
            width=size[2],
            height=size[0],
            depth=size[1],
            opacity=opacity,
        ),
        to_Conv(
            name=f"ccr_{name}",
            offset="(0,0,0)",
            to=f"(ccr_res_{name}-east)",
            s_filer=str(s_filer),
            n_filer=str(n_filer),
            width=size[2],
            height=size[0],
            depth=size[1],
        ),
        to_ConvRes(
            name=f"ccr_res_c_{name}",
            offset="(0,0,0)",
            to=f"(ccr_{name}-east)",
            s_filer=str(s_filer),
            n_filer=str(n_filer),
            width=size[2],
            height=size[0],
            depth=size[1],
            opacity=opacity,
        ),
        to_Conv(
            name=f"{top}",
            offset="(0,0,0)",
            to=f"(ccr_res_c_{name}-east)",
            s_filer=str(s_filer),
            n_filer=str(n_filer),
            width=size[2],
            height=size[0],
            depth=size[1],
        ),
        to_connection(f"{botton}", f"unpool_{name}"),
    ]


def block_Res(
    num, name, botton, top, s_filer=256, n_filer=64, offset="(0,0,0)", size=(32, 32, 3.5), opacity=0.5
):
    lys = []
    layers = [*[f"{name}_{i}" for i in range(num - 1)], top]
    for name in layers:
        ly = [
            to_Conv(
                name=f"{name}",
                offset=offset,
                to=f"({botton}-east)",
                s_filer=str(s_filer),
                n_filer=str(n_filer),
                width=size[2],
                height=size[0],
                depth=size[1],
            ),
            to_connection(f"{botton}", f"{name}"),
        ]
        botton = name
        lys += ly

    lys += [
        to_skip(of=layers[1], to=layers[-2], pos=1.25),
    ]
    return lys
