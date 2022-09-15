"""Convert VGG16 weights from TensorFlow (1.x) format to PyTorch.

The input path is _not_ the actual path to the checkpoint files. It is the stem
of these files. For example, if the checkpoint files are "foo.ckpt.index", pass the
name "foo.ckpt" to this script.

This script requires TensorFlow 2.x. Install it with 'pip install tensorflow-cpu'.

The original TIL model is at
https://stonybrookmedicine.app.box.com/v/til-results-new-model/folder/128593971923.

It was implemented in TF Slim. This script converts those TF Slim weights to PyTorch.
"""

import argparse
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Do not print INFO messages

import tensorflow as tf
import torch
import torchvision

if not tf.__version__.startswith("2."):
    raise EnvironmentError("TensorFlow 2.x must be installed.")

tf_to_pytorch_layers = [
    ("vgg_16/conv1/conv1_1", "features.0"),
    ("vgg_16/conv1/conv1_2", "features.2"),
    ("vgg_16/conv2/conv2_1", "features.5"),
    ("vgg_16/conv2/conv2_2", "features.7"),
    ("vgg_16/conv3/conv3_1", "features.10"),
    ("vgg_16/conv3/conv3_2", "features.12"),
    ("vgg_16/conv3/conv3_3", "features.14"),
    ("vgg_16/conv4/conv4_1", "features.17"),
    ("vgg_16/conv4/conv4_2", "features.19"),
    ("vgg_16/conv4/conv4_3", "features.21"),
    ("vgg_16/conv5/conv5_1", "features.24"),
    ("vgg_16/conv5/conv5_2", "features.26"),
    ("vgg_16/conv5/conv5_3", "features.28"),
    ("vgg_16/fc6", "classifier.0"),
    ("vgg_16/fc7", "classifier.3"),
    ("vgg_16/fc8", "classifier.6"),
]


def convert_tf_to_pytorch(input_path, output_path, num_classes: int):
    try:
        ckpt = tf.train.load_checkpoint(input_path)
    except tf.errors.DataLossError:
        raise RuntimeError(
            "Error: could not load checkpoint. Did you pass in the stem of the path?"
            "Pass in the path without '.index' or '.meta' or '.data-00000-of-00001'."
        )
    new_state_dict = {}
    for tf_prefix, torch_prefix in tf_to_pytorch_layers:
        tf_weights = f"{tf_prefix}/weights"
        tf_biases = f"{tf_prefix}/biases"
        torch_weights = f"{torch_prefix}.weight"
        torch_biases = f"{torch_prefix}.bias"
        tf_weight_array = ckpt.get_tensor(tf_weights)
        tf_bias_array = ckpt.get_tensor(tf_biases)
        if "conv" in tf_weights:
            tf_weight_array = tf_weight_array.transpose([3, 2, 0, 1])
        elif "fc" in tf_weights:
            if tf_weights == "vgg_16/fc6/weights":
                # [7, 7, 512, 4096] -> [25088, 4096]
                tf_weight_array = tf_weight_array.reshape((25088, 4096))
            # E.g., go from shape [1, 1, 4096, 1000] to [1000, 4096]
            tf_weight_array = tf_weight_array.squeeze().T
        new_state_dict[torch_weights] = torch.from_numpy(tf_weight_array)
        new_state_dict[torch_biases] = torch.from_numpy(tf_bias_array)

    # Test that conversion was (probably) done correctly.
    true_model = torchvision.models.vgg16()
    true_model.classifier[6] = torch.nn.Linear(4096, num_classes)
    if true_model.state_dict().keys() != new_state_dict.keys():
        raise RuntimeError(
            "Something went wrong... converted model keys do not match TorchVision"
            " VGG16 keys."
        )
    true_state_dict = true_model.state_dict()
    for true_k, new_k in zip(true_state_dict, new_state_dict):
        true_shape = true_state_dict[true_k].shape
        new_shape = new_state_dict[new_k].shape
        if true_shape != new_shape:
            raise ValueError(
                "Shape mismatch between converted parameters and reference parameters."
            )

    torch.save(new_state_dict, output_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", help="Path to the VGG16 tensorflow checkpoint.")
    p.add_argument("output")
    p.add_argument(
        "--num-classes", type=int, default=2, help="Number of output classes."
    )
    args = p.parse_args()
    # We do not expect the input "path" to exist, because it should be the stem.
    if Path(args.input).exists():
        raise ValueError(
            "Input path exists. Instead of passing the full path, pass in the stem"
            " (i.e., without '.index' or '.meta' or '.data-00000-of-00001'."
        )
    convert_tf_to_pytorch(args.input, args.output, args.num_classes)
