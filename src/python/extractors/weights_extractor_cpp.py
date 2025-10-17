from abc import ABC, abstractmethod
import argparse
import os

from brevitas.nn import QuantConv2d
import numpy as np
import torch
import yaml

from network import networks


def float_to_fixed_array(values, scale, dtype=np.int8):
    values = np.asarray(values, dtype=np.float32)
    scale = np.asarray(scale, dtype=np.float32)
    q = np.round(values / scale)
    return q.astype(dtype)


def bit_width_to_dtype(bit_width, signed=True):
    """
    Returns a NumPy integer dtype corresponding to the given bit width.
    Parameters:
        bit_width (int): Bit width of the desired integer type (8, 16, 32, 64).
        signed (bool): Whether to use signed integers. Default is True.
    Returns:
        np.dtype: Corresponding NumPy dtype (e.g., np.int8, np.uint16).
    Raises:
        ValueError: If the bit width is not supported.
    """
    dtype_map = {
        8:  (np.int8,  np.uint8),
        16: (np.int16, np.uint16),
        32: (np.int32, np.uint32),
        64: (np.int64, np.uint64),
    }

    if bit_width not in dtype_map:
        raise ValueError(f"Unsupported bit width: {bit_width}. Must be one of: {list(dtype_map.keys())}")
    return dtype_map[bit_width][0] if signed else dtype_map[bit_width][1]



def load_yaml(file_path) -> dict:
    with open(file_path) as file:
        return yaml.safe_load(file)


def setcache(m):
    m.cache_inference_quant_bias = True


def parse_args():
    parser = argparse.ArgumentParser()

    # base
    parser.add_argument(
        "--model_dir",
        type=str,
        help="Trained model directory (needs files: model.pth, config.yml)",
        required=True,
    )
    return vars(parser.parse_args())


def load_model(config, model_pth, device):
    """
    Load the trained model.
    """
    model = networks.get_model(model_name=config["model"], config=config).to(device)
    model.apply(setcache)
    state_dict = torch.load(model_pth, map_location=device, weights_only=True)
    model.load_state_dict(state_dict=state_dict, strict=False)
    return model.eval()


def generate_input(config, device):
    C = config["input_channels"]
    H = config["input_h"]
    W = config["input_w"]
    return torch.zeros(1, C, H, W).to(device)


def array_2_string_cstyle(arr, idx, indent=4):
    """
    Convert a NumPy array to a nicely formatted C-style initializer (just the body, no declaration),
    with a C comment indicating the shape.
    """
    def format_array(a, level=1):
        spacing = ' ' * (indent * level)
        if a.ndim == 1:
            elements = ', '.join(f"{x:2}" for x in a)
            return spacing + "{ " + elements + "}"

        inner = ',\n'.join(format_array(sub, level+1) for sub in a)
        return spacing + "{\n" + inner + "\n" + spacing + "}"

    c_comment = f"/* idx: {idx}, shape: {arr.shape} */\n"
    return c_comment + format_array(arr, 0) + ","


def delete_if_found(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)



class ParametersCppBase(ABC):
    def __init__(self, output_dir, weight_filename, bias_filename):
        self.weight_name = os.path.splitext(weight_filename)[0]
        self.bias_name = os.path.splitext(bias_filename)[0]
        os.makedirs(output_dir, exist_ok=True)

        self.weigths_file = os.path.join(output_dir, weight_filename)
        if os.path.exists(self.weigths_file):
            os.remove(self.weigths_file)

        self.bias_file = os.path.join(output_dir, bias_filename)
        if os.path.exists(self.bias_file):
            os.remove(self.bias_file)

        self.w_counter = 0
        self.b_counter = 0
        self.w_max_shape = None
        self.b_max_shape = None

    @abstractmethod
    def handle_weights(self, fixed_weights):
        """Transform the weights into the desired shape."""

    def _array_2_string_cstyle(self, arr, idx, indent=4):
        """
        Convert a NumPy array to a nicely formatted C-style initializer,
        with a C comment indicating index and shape.
        """
        def format_array(a, level=1):
            spacing = ' ' * (indent * level)
            if a.ndim == 1:
                elements = ', '.join(f"{x:2}" for x in a)
                return spacing + "{ " + elements + "}"

            inner = ',\n'.join(format_array(sub, level+1) for sub in a)
            return spacing + "{\n" + inner + "\n" + spacing + "}"

        c_comment = f"\n\n/* idx: {idx}, shape: {arr.shape} */\n"
        return c_comment + format_array(arr, 1) + ","

    def write_weights(self, fixed_weights):
        handled_weights = self.handle_weights(fixed_weights)
        c_string = self._array_2_string_cstyle(handled_weights, self.w_counter)
        with open(self.weigths_file, "a") as f:
            if self.w_counter == 0:
                self.w_max_shape = handled_weights.shape
                brackets = "[]" * handled_weights.ndim
                f.write("// Auto-generated from quantized weights model\n")
                f.write("// See end of map declaration to find max map size\n")
                f.write(f"const static int {self.weight_name}[]{brackets} = {{")
            else:
                self.w_max_shape = tuple(max(a, b) for a, b in zip(self.w_max_shape, handled_weights.shape, strict=False))
            f.write(c_string)

        self.w_counter += 1

    def write_bias(self, fixed_bias):
        c_string = self._array_2_string_cstyle(fixed_bias, self.w_counter)
        with open(self.bias_file, "a") as f:
            if self.b_counter == 0:
                self.b_max_shape = fixed_bias.shape
                brackets = "[]" * fixed_bias.ndim
                f.write("// Auto-generated from quantized weights model\n")
                f.write("// See end of map declaration to find max map size\n")
                f.write(f"const static int {self.bias_name}[]{brackets} = {{")
            else:
                self.b_max_shape = tuple(max(a, b) for a, b in zip(self.b_max_shape, fixed_bias.shape, strict=False))
            f.write(c_string)

        self.b_counter += 1

    def cleanup(self):
        if self.w_counter > 0:
            with open(self.weigths_file, "a") as f:
                f.write("\n};\n")
                dims_brackets = ''.join(f'[{d}]' for d in self.w_max_shape)
                f.write(f"// [{self.w_counter}]{dims_brackets} \n")

        if self.b_counter > 0:
            with open(self.bias_file, "a") as f:
                f.write("\n};\n")
                dims_brackets = ''.join(f'[{d}]' for d in self.b_max_shape)
                f.write(f"// [{self.w_counter}]{dims_brackets} \n")

class ParametersPWCpp(ParametersCppBase):
    def __init__(self, output_dir):
        super().__init__(output_dir, "weightsPW.h", "biasPW.h")

    def handle_weights(self, fixed_weights):
        # (outC, inC, 1, 1) -> (outC, inC)
        return np.squeeze(fixed_weights, axis=(2, 3))


class ParametersDWCpp(ParametersCppBase):
    def __init__(self, output_dir):
        super().__init__(output_dir, "weightsDW.h", "biasDW.h")

    def handle_weights(self, fixed_weights):
        # (outC, 1, kH, kW) -> (kH, kW, outC)
        return np.squeeze(fixed_weights, axis=1).transpose(1, 2, 0)

class Parameters3DCpp(ParametersCppBase):
    def __init__(self, output_dir):
        super().__init__(output_dir, "weights3D.h", "bias3D.h")

    def handle_weights(self, fixed_weights):
        # (outC, inC, kH, kW) -> (kH, kW, outC, inC)
        return fixed_weights.transpose(2, 3, 0, 1)

# class ParametersG2Cpp(ParametersCppBase):
#     def __init__(self, output_dir):
#         super().__init__(output_dir, "weightsG2.h", "biasG2.h")

#     def handle_weights(self, fixed_weights):
#         # (outC, inC, kH, kW) -> (kH, kW, outC, inC)
#         return  fixed_weights.transpose(2, 3, 0, 1)



def extract_weights_and_bias(model, output_dir, device):
    os.makedirs(output_dir, exist_ok=True)
    id = 0

    weightsDWCpp = ParametersDWCpp(output_dir)
    weightsPWCpp = ParametersPWCpp(output_dir)
    # weightsG2Cpp = ParametersG2Cpp(output_dir)
    weights3DCpp = Parameters3DCpp(output_dir)
    weights_cpp_class : ParametersCppBase = None

    for name, module in model.named_modules():
        base_name = f"{id}_{name}_{module.__class__.__name__}"
        description_file_name = f"{base_name}.txt"
        debug_file_name =  f"{base_name}.values"
        bin_file_name = f"{base_name}.bin"
        description_file = os.path.join(output_dir, description_file_name)
        bin_file = os.path.join(output_dir, bin_file_name)
        debug_file = os.path.join(output_dir, debug_file_name)

        delete_if_found(description_file)
        delete_if_found(bin_file)
        delete_if_found(debug_file)

        if isinstance(module, QuantConv2d):

            if module.groups == module.in_channels and module.in_channels == module.out_channels:
                weights_cpp_class = weightsDWCpp
            elif module.kernel_size == (1, 1) and module.groups == 1:
                weights_cpp_class = weightsPWCpp
            # elif module.groups == 2:
                # weights_cpp_class = weightsG2Cpp
            else:
                weights_cpp_class = weights3DCpp


            # === Weights ===
            if hasattr(module, "weight") and hasattr(module, "weight_quant"):
                id+=1
                quant_tensor = module.quant_weight()
                scale = getattr(quant_tensor, "scale", None)
                scale = scale.item() if hasattr(scale, 'item') else scale
                signed = getattr(quant_tensor, "signed_t", None)
                weights = getattr(quant_tensor, "value", None)
                bit_width = getattr(quant_tensor, "bit_width", None)
                dtype = bit_width_to_dtype(int(bit_width), signed)

                weights_np = weights.detach().to(device).numpy()
                fixed_values = float_to_fixed_array(weights_np, scale, dtype)

                with open(description_file, "a") as f:
                    f.write("#" * 80 + "\n")
                    f.write(f"[Weight] Layer: {name} ({module.__class__.__name__})\n")
                    f.write(f"  Shape: {tuple(quant_tensor.shape)}\n")
                    f.write(f"  Bit Width: {bit_width.item()}\n")
                    f.write(f"  Signed: {signed}\n")
                    f.write(f"  Scale: {scale}\n")
                    f.write(
                        f"  Sample Weights: {np.array2string(weights_np[:10], precision=5)}...\n"
                    )
                    f.write("-" * 80 + "\n")

                with open(debug_file, "a") as f:
                    f.write(np.array2string(weights_np, precision=5))
                    f.write("\nFixed Point: \n" + np.array2string(fixed_values))

                with open(bin_file, "ab") as f:
                    f.write(fixed_values.tobytes())

                # ----------------------------------C FILE STORE ------------------------------------
                weights_cpp_class.write_weights(fixed_values)

            # === Bias ===
            if hasattr(module, "bias_quant"):
                quant_tensor = module.quant_bias()
                scale = getattr(quant_tensor, "scale", None)
                scale = scale.item() if hasattr(scale, 'item') else scale
                signed = getattr(quant_tensor, "signed_t", None)
                bias = getattr(quant_tensor, "value", None)
                bit_width = getattr(quant_tensor, "bit_width", None)

                bias_np = bias.detach().to(device).numpy()
                array_flat = bias_np.flatten(order="C")

                with open(description_file, "a") as f:
                    f.write(f"[Bias] Layer: {name} ({module.__class__.__name__})\n")
                    f.write(f"  Shape: {tuple(quant_tensor.shape)}\n")
                    f.write(f"  Bit Width: {bit_width.item()}\n")
                    f.write(f"  Signed: {signed}\n")
                    f.write(f"  Scale: {scale}\n")
                    f.write(
                        f"  Sample Bias: {np.array2string(array_flat[:10], precision=5)}...\n"
                    )
                    f.write("#" * 80 + "\n\n")

                dtype = bit_width_to_dtype(int(bit_width), signed)
                fixed_values = float_to_fixed_array(array_flat, scale, dtype)

                with open(debug_file, "a") as f:
                    f.write("\n" + "-" * 80 + "\n")
                    f.write(np.array2string(array_flat, precision=5))
                    f.write("\nFixed Point: \n" + np.array2string(fixed_values))

                with open(bin_file, "ab") as f:
                    f.write(fixed_values.tobytes())

                # ----------------------------------C FILE STORE ------------------------------------
                weights_cpp_class.write_bias(fixed_values)


    # DONT FORGET
    weightsDWCpp.cleanup()
    weightsPWCpp.cleanup()
    weights3DCpp.cleanup()
    # weightsG2Cpp.cleanup()


def weight_extractor(device, model_dir):
    config = load_yaml(os.path.join(model_dir, "config.yml"))
    model_pth = os.path.join(model_dir, "model.pth")
    model = load_model(config, model_pth, device)

    output_dir = os.path.join(model_dir, "weights")

    input_ = generate_input(config, device)
    model(input_)  # Run a dummy inference first
    extract_weights_and_bias(model, output_dir, device)

