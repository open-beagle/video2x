from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference


def tensor_shape(model: onnx.ModelProto, name: str) -> tuple[int, list[int | str]]:
    values = list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output)
    for value in values:
        if value.name != name:
            continue
        tensor_type = value.type.tensor_type
        elem_type = tensor_type.elem_type
        shape: list[int | str] = []
        for dim in tensor_type.shape.dim:
            shape.append(dim.dim_value or dim.dim_param)
        return elem_type, shape
    raise ValueError(f"shape not found for tensor: {name}")


def initializer_map(model: onnx.ModelProto) -> dict[str, onnx.TensorProto]:
    return {initializer.name: initializer for initializer in model.graph.initializer}


def producer_map(model: onnx.ModelProto) -> dict[str, tuple[int, onnx.NodeProto]]:
    producers: dict[str, tuple[int, onnx.NodeProto]] = {}
    for index, node in enumerate(model.graph.node):
        for output in node.output:
            producers[output] = (index, node)
    return producers


def find_srvgg_tail(model: onnx.ModelProto) -> tuple[str, onnx.NodeProto, onnx.NodeProto, onnx.NodeProto]:
    if len(model.graph.output) != 1:
        raise ValueError("expected one graph output")

    producers = producer_map(model)
    output_name = model.graph.output[0].name
    _, add = producers[output_name]
    if add.op_type != "Add":
        raise ValueError(f"expected graph output producer Add, got {add.op_type}")

    depth = None
    resize = None
    for input_name in add.input:
        producer = producers.get(input_name)
        if not producer:
            continue
        node = producer[1]
        if node.op_type == "DepthToSpace":
            depth = node
        elif node.op_type == "Resize":
            resize = node
    if depth is None or resize is None:
        raise ValueError("expected Add inputs from DepthToSpace and Resize")

    conv_producer = producers.get(depth.input[0])
    if not conv_producer or conv_producer[1].op_type != "Conv":
        raise ValueError("expected DepthToSpace input producer Conv")
    final_conv = conv_producer[1]
    tail_input = final_conv.input[0]
    return tail_input, final_conv, depth, resize


def prune_initializers(model: onnx.ModelProto) -> None:
    needed: set[str] = set()
    for node in model.graph.node:
        needed.update(node.input)
    kept = [initializer for initializer in model.graph.initializer if initializer.name in needed]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)


def main() -> int:
    parser = argparse.ArgumentParser(description="Split SRVGG final Conv/PixelShuffle tail from an ONNX graph.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tail-weights", type=Path, required=True)
    parser.add_argument(
        "--split-at",
        choices=["pre-conv", "post-conv"],
        default="pre-conv",
        help="pre-conv outputs prelu_32 and saves final conv weights; post-conv outputs Conv(64->48).",
    )
    args = parser.parse_args()

    model = onnx.load(args.input, load_external_data=True)
    inferred = shape_inference.infer_shapes(model)
    tail_input, final_conv, depth, resize = find_srvgg_tail(inferred)

    initializers = initializer_map(model)
    weight_name = final_conv.input[1]
    bias_name = final_conv.input[2] if len(final_conv.input) > 2 else ""
    if weight_name not in initializers:
        raise ValueError(f"final Conv weight initializer not found: {weight_name}")
    weight = numpy_helper.to_array(initializers[weight_name])
    bias = numpy_helper.to_array(initializers[bias_name]) if bias_name else np.zeros((weight.shape[0],), dtype=weight.dtype)

    blocksize = 0
    mode = "CRD"
    for attr in depth.attribute:
        if attr.name == "blocksize":
            blocksize = int(helper.get_attribute_value(attr))
        elif attr.name == "mode":
            mode = helper.get_attribute_value(attr).decode()
    if blocksize != 4:
        raise ValueError(f"expected DepthToSpace blocksize=4, got {blocksize}")

    producers = producer_map(model)
    if args.split_at == "post-conv":
        split_output = final_conv.output[0]
        split_type, split_shape = tensor_shape(inferred, split_output)
        split_producer = producers.get(split_output)
        if not split_producer:
            raise ValueError(f"split output producer not found: {split_output}")
        keep_count = split_producer[0] + 1
    else:
        split_output = tail_input
        split_type, split_shape = tensor_shape(inferred, split_output)
        split_producer = producers.get(split_output)
        if not split_producer:
            raise ValueError(f"split output producer not found: {split_output}")
        keep_count = split_producer[0] + 1
    if split_type not in (TensorProto.FLOAT16, TensorProto.FLOAT):
        raise ValueError(f"unsupported split output type: {split_type}")

    del model.graph.node[keep_count:]
    del model.graph.output[:]
    model.graph.output.extend([helper.make_tensor_value_info(split_output, split_type, split_shape)])
    prune_initializers(model)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.tail_weights.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(
        model,
        args.output,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=args.output.name + ".data",
        size_threshold=1024,
    )
    np.savez(
        args.tail_weights,
        weight=weight,
        bias=bias,
        blocksize=np.array([blocksize], dtype=np.int32),
        mode=np.array([mode]),
        tail_input=np.array([tail_input]),
        split_at=np.array([args.split_at]),
        split_output=np.array([split_output]),
        resize_mode=np.array([resize.attribute[0].name if resize.attribute else ""]),
    )

    print(f"input={args.input}")
    print(f"output={args.output}")
    print(f"tail_weights={args.tail_weights}")
    print(f"split_at={args.split_at}")
    print(f"split_output={split_output} shape={split_shape} elem_type={split_type}")
    print(f"tail_input={tail_input}")
    print(f"final_conv_weight={weight_name} shape={list(weight.shape)} dtype={weight.dtype}")
    print(f"final_conv_bias={bias_name} shape={list(bias.shape)} dtype={bias.dtype}")
    print(f"depth_to_space=blocksize:{blocksize} mode:{mode}")
    print(f"kept_nodes={len(model.graph.node)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
