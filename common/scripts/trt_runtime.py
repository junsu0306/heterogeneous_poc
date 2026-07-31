"""Phase 1 shared TensorRT build/run helpers.

No pycuda / cuda-python is installed on this machine, and none is needed:
torch CUDA tensors already expose device pointers via `.data_ptr()`, which
is all TensorRT's calibrator and execution-context APIs require. This
module reuses that pattern (first proven out in scripts/build_implicit.py)
so every Phase 1 script builds/runs engines the same way instead of each
reimplementing it.

Convention (poc_implementation_handoff_v2.md §0): seed=42, diffs computed
in float64, results appended to results/p1.json, one key per gate.
"""
import json
import os
import random

import numpy as np
import tensorrt as trt
import torch

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------
# INT8 calibrators fed by an in-memory list of samples (no dataset on disk).
# Two variants because the two experiments need different scale semantics:
#   - EntropyListCalibrator: KL-divergence histogram search, same algorithm
#     P0 used for ResNet-50 (scripts/build_implicit.py). Good default for
#     "does any divergence appear at all" style checks (1.0).
#   - MinMaxListCalibrator: scale = max_abs_seen / 127, no histogram search.
#     Needed whenever the scale must be analytically predictable from the
#     calibration data itself (1.1, 1.2).
# --------------------------------------------------------------------------
class _ListCalibratorMixin:
    def _init_samples(self, samples, cache_file):
        self.samples = [torch.as_tensor(s, dtype=torch.float32).cuda().contiguous() for s in samples]
        self.cache_file = cache_file
        self.idx = 0

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        if self.idx >= len(self.samples):
            return None
        ptr = int(self.samples[self.idx].data_ptr())
        self.idx += 1
        return [ptr]

    def read_calibration_cache(self):
        if self.cache_file and os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        if self.cache_file:
            with open(self.cache_file, "wb") as f:
                f.write(cache)


class EntropyListCalibrator(_ListCalibratorMixin, trt.IInt8EntropyCalibrator2):
    def __init__(self, samples, cache_file=None):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self._init_samples(samples, cache_file)


class MinMaxListCalibrator(_ListCalibratorMixin, trt.IInt8MinMaxCalibrator):
    def __init__(self, samples, cache_file=None):
        trt.IInt8MinMaxCalibrator.__init__(self)
        self._init_samples(samples, cache_file)


# --------------------------------------------------------------------------
# Build / run
# --------------------------------------------------------------------------
def build_int8_engine(
    onnx_path,
    device,
    calibrator,
    allow_gpu_fallback=True,
    force_layer_int8=False,
    detailed_inspector=False,
    dla_core=0,
):
    """device: 'gpu' or 'dla'. Returns serialized engine bytes, or None on failure."""
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as f:
        ok = parser.parse(f.read())
    if not ok:
        errs = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"onnx parse failed:\n{errs}")

    # Under strict layer-INT8 constraints, give every network output an
    # explicit FP32 identity/reformat layer. A tiny Conv-only GPU graph cannot
    # otherwise satisfy "INT8 Conv output" and "FP32 network binding" at once.
    output_reformat_names = set()
    if force_layer_int8:
        original_outputs = [
            network.get_output(i) for i in range(network.num_outputs)
        ]
        for output_index, tensor in enumerate(original_outputs):
            original_name = tensor.name
            network.unmark_output(tensor)
            tensor.name = f"{original_name}__int8_internal"
            reformat = network.add_identity(tensor)
            reformat.name = f"__v13_fp32_output_reformat_{output_index}"
            reformat.precision = trt.DataType.FLOAT
            reformat.set_output_type(0, trt.DataType.FLOAT)
            converted = reformat.get_output(0)
            converted.name = original_name
            network.mark_output(converted)
            converted.dtype = trt.DataType.FLOAT
            output_reformat_names.add(reformat.name)

    # Pin every network output to float. Matters for subgraphs cut close to
    # the input (e.g. boundary_divergence.py heads): TRT can fail to find an
    # INT8-internal/float-output format for a tiny fused block otherwise.
    for i in range(network.num_outputs):
        network.get_output(i).dtype = trt.DataType.FLOAT

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = calibrator
    if detailed_inspector:
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    if force_layer_int8:
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        # Shape/control/constant layers can carry INT32/INT64 metadata and
        # cannot legally be assigned INT8 precision. Pin only arithmetic
        # layers whose tensors form the activation path.
        strict_layer_types = {
            trt.LayerType.ACTIVATION,
            trt.LayerType.CONCATENATION,
            trt.LayerType.CONVOLUTION,
            trt.LayerType.DECONVOLUTION,
            trt.LayerType.EINSUM,
            trt.LayerType.ELEMENTWISE,
            trt.LayerType.GRID_SAMPLE,
            trt.LayerType.IDENTITY,
            trt.LayerType.LRN,
            trt.LayerType.MATRIX_MULTIPLY,
            trt.LayerType.NORMALIZATION,
            trt.LayerType.PADDING,
            trt.LayerType.PARAMETRIC_RELU,
            trt.LayerType.POOLING,
            trt.LayerType.REDUCE,
            trt.LayerType.RESIZE,
            trt.LayerType.SCALE,
            trt.LayerType.SELECT,
            trt.LayerType.SLICE,
            trt.LayerType.SOFTMAX,
            trt.LayerType.UNARY,
        }
        for layer_index in range(network.num_layers):
            layer = network.get_layer(layer_index)
            if (
                layer.name in output_reformat_names
                or layer.type not in strict_layer_types
            ):
                continue
            layer.precision = trt.DataType.INT8
            for output_index in range(layer.num_outputs):
                output = layer.get_output(output_index)
                if (
                    output is not None
                    and not output.is_shape_tensor
                    and output.dtype
                    in {
                        trt.DataType.FLOAT,
                        trt.DataType.HALF,
                        trt.DataType.INT8,
                    }
                ):
                    layer.set_output_type(output_index, trt.DataType.INT8)

    if device == "dla":
        if allow_gpu_fallback:
            config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
        config.default_device_type = trt.DeviceType.DLA
        config.DLA_core = int(dla_core)
    elif device != "gpu":
        raise ValueError(f"device must be 'gpu' or 'dla', got {device!r}")

    return builder.build_serialized_network(network, config)


def load_engine(serialized_or_path):
    """Accepts a path, or whatever build_int8_engine() returned (bytes or
    the trt.IHostMemory buffer build_serialized_network gives back)."""
    runtime = trt.Runtime(TRT_LOGGER)
    if isinstance(serialized_or_path, (str, os.PathLike)):
        with open(serialized_or_path, "rb") as f:
            return runtime.deserialize_cuda_engine(f.read())
    return runtime.deserialize_cuda_engine(bytes(serialized_or_path))


def run_engine(engine, input_array):
    """Runs a single-input/single-output engine on `input_array` (numpy).

    Returns the output as a numpy float32 array. Buffers are torch CUDA
    tensors bound to TensorRT via set_tensor_address -- no pycuda/cuda-python.
    """
    ctx = engine.create_execution_context()
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    in_name = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
    out_name = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)

    x = torch.as_tensor(input_array, dtype=torch.float32).cuda().contiguous()
    ctx.set_input_shape(in_name, tuple(x.shape))
    ctx.set_tensor_address(in_name, x.data_ptr())

    out_shape = tuple(ctx.get_tensor_shape(out_name))
    y = torch.empty(out_shape, dtype=torch.float32, device="cuda")
    ctx.set_tensor_address(out_name, y.data_ptr())

    # `x.cuda()` is enqueued on PyTorch's current stream. Running TensorRT
    # immediately on an unrelated stream creates a race in which TensorRT
    # can consume a partially copied input. Use the same stream so the H2D
    # copy, TensorRT execution, and output copy are correctly ordered.
    stream = torch.cuda.current_stream()
    ok = ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    if not ok:
        raise RuntimeError("execute_async_v3 failed")
    return y.cpu().numpy()


def run_engine_multi_output(engine, input_array):
    """Like run_engine(), but for engines with >1 output tensor (e.g. a
    network exported with intermediate taps as extra ONNX graph outputs).
    Returns {output_name: numpy_array}, in engine binding order."""
    ctx = engine.create_execution_context()
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    in_name = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
    out_names = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]

    x = torch.as_tensor(input_array, dtype=torch.float32).cuda().contiguous()
    ctx.set_input_shape(in_name, tuple(x.shape))
    ctx.set_tensor_address(in_name, x.data_ptr())

    out_bufs = {}
    for out_name in out_names:
        out_shape = tuple(ctx.get_tensor_shape(out_name))
        y = torch.empty(out_shape, dtype=torch.float32, device="cuda")
        ctx.set_tensor_address(out_name, y.data_ptr())
        out_bufs[out_name] = y

    stream = torch.cuda.current_stream()
    ok = ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    if not ok:
        raise RuntimeError("execute_async_v3 failed")
    return {name: buf.cpu().numpy() for name, buf in out_bufs.items()}


class EngineRunner:
    """Like run_engine(), but keeps one execution context + device buffers
    alive across many calls instead of recreating them every time. Matters
    once you're running hundreds/thousands of single-image inferences (e.g.
    CA/ASR evaluation over an eval set) -- context creation and device
    malloc are not free, and run_engine() pays that cost on every call."""

    def __init__(self, engine):
        self.engine = engine
        self.ctx = engine.create_execution_context()
        names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        self.in_name = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self.out_name = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
        # Inputs are copied by PyTorch on its current stream in run(). Keep
        # TensorRT on that stream as well; a private stream here would need
        # explicit events before every execution.
        self.stream = torch.cuda.current_stream()
        self._out_buf = None
        self._out_shape = None

    def run(self, input_array):
        x = torch.as_tensor(input_array, dtype=torch.float32).cuda().contiguous()
        self.ctx.set_input_shape(self.in_name, tuple(x.shape))
        self.ctx.set_tensor_address(self.in_name, x.data_ptr())

        out_shape = tuple(self.ctx.get_tensor_shape(self.out_name))
        if out_shape != self._out_shape:
            self._out_buf = torch.empty(out_shape, dtype=torch.float32, device="cuda")
            self._out_shape = out_shape
        self.ctx.set_tensor_address(self.out_name, self._out_buf.data_ptr())

        ok = self.ctx.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        if not ok:
            raise RuntimeError("execute_async_v3 failed")
        return self._out_buf.cpu().numpy()


def diff_stats(a, b):
    """float64 diff per common regs (poc_implementation_handoff_v2.md §0)."""
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    max_abs_diff = float(np.max(np.abs(a64 - b64)))
    mean_abs_diff = float(np.mean(np.abs(a64 - b64)))
    mean_abs_ref = float(np.mean(np.abs(a64)))
    rel_diff = mean_abs_diff / (mean_abs_ref + 1e-12)
    return {"max_abs_diff": max_abs_diff, "mean_abs_diff": mean_abs_diff, "rel_diff": rel_diff}


def update_results(key, value, path="results/p1.json"):
    data = {}
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
    data[key] = value
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[results] wrote {path} :: {key}")
