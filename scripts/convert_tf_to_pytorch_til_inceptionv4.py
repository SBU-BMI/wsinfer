"""Convert InceptionV4 weights from TensorFlow (1.x) format to PyTorch.

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

inceptionv4 = None
try:
    from wsinfer.modellib.inceptionv4_no_batchnorm import inceptionv4
except ImportError:
    pass

if not tf.__version__.startswith("2."):
    raise EnvironmentError("TensorFlow 2.x must be installed.")

tf_to_pytorch_layers = [
    # Basic convs
    ("InceptionV4/Conv2d_1a_3x3", "features.0.conv"),
    ("InceptionV4/Conv2d_2a_3x3", "features.1.conv"),
    ("InceptionV4/Conv2d_2b_3x3", "features.2.conv"),
    # Mixed_3a
    ("InceptionV4/Mixed_3a/Branch_1/Conv2d_0a_3x3", "features.3.conv.conv"),
    # Mixed_4a
    ("InceptionV4/Mixed_4a/Branch_0/Conv2d_0a_1x1", "features.4.branch0.0.conv"),
    ("InceptionV4/Mixed_4a/Branch_0/Conv2d_1a_3x3", "features.4.branch0.1.conv"),
    ("InceptionV4/Mixed_4a/Branch_1/Conv2d_0a_1x1", "features.4.branch1.0.conv"),
    ("InceptionV4/Mixed_4a/Branch_1/Conv2d_0b_1x7", "features.4.branch1.1.conv"),
    ("InceptionV4/Mixed_4a/Branch_1/Conv2d_0c_7x1", "features.4.branch1.2.conv"),
    ("InceptionV4/Mixed_4a/Branch_1/Conv2d_1a_3x3", "features.4.branch1.3.conv"),
    # Mixed_5a
    ("InceptionV4/Mixed_5a/Branch_0/Conv2d_1a_3x3", "features.5.conv.conv"),
    # InceptionA (1)
    ("InceptionV4/Mixed_5b/Branch_0/Conv2d_0a_1x1", "features.6.branch0.conv"),
    ("InceptionV4/Mixed_5b/Branch_1/Conv2d_0a_1x1", "features.6.branch1.0.conv"),
    ("InceptionV4/Mixed_5b/Branch_1/Conv2d_0b_3x3", "features.6.branch1.1.conv"),
    ("InceptionV4/Mixed_5b/Branch_2/Conv2d_0a_1x1", "features.6.branch2.0.conv"),
    ("InceptionV4/Mixed_5b/Branch_2/Conv2d_0b_3x3", "features.6.branch2.1.conv"),
    ("InceptionV4/Mixed_5b/Branch_2/Conv2d_0c_3x3", "features.6.branch2.2.conv"),
    ("InceptionV4/Mixed_5b/Branch_3/Conv2d_0b_1x1", "features.6.branch3.1.conv"),
    # InceptionA (2)
    ("InceptionV4/Mixed_5c/Branch_0/Conv2d_0a_1x1", "features.7.branch0.conv"),
    ("InceptionV4/Mixed_5c/Branch_1/Conv2d_0a_1x1", "features.7.branch1.0.conv"),
    ("InceptionV4/Mixed_5c/Branch_1/Conv2d_0b_3x3", "features.7.branch1.1.conv"),
    ("InceptionV4/Mixed_5c/Branch_2/Conv2d_0a_1x1", "features.7.branch2.0.conv"),
    ("InceptionV4/Mixed_5c/Branch_2/Conv2d_0b_3x3", "features.7.branch2.1.conv"),
    ("InceptionV4/Mixed_5c/Branch_2/Conv2d_0c_3x3", "features.7.branch2.2.conv"),
    ("InceptionV4/Mixed_5c/Branch_3/Conv2d_0b_1x1", "features.7.branch3.1.conv"),
    # InceptionA (3)
    ("InceptionV4/Mixed_5d/Branch_0/Conv2d_0a_1x1", "features.8.branch0.conv"),
    ("InceptionV4/Mixed_5d/Branch_1/Conv2d_0a_1x1", "features.8.branch1.0.conv"),
    ("InceptionV4/Mixed_5d/Branch_1/Conv2d_0b_3x3", "features.8.branch1.1.conv"),
    ("InceptionV4/Mixed_5d/Branch_2/Conv2d_0a_1x1", "features.8.branch2.0.conv"),
    ("InceptionV4/Mixed_5d/Branch_2/Conv2d_0b_3x3", "features.8.branch2.1.conv"),
    ("InceptionV4/Mixed_5d/Branch_2/Conv2d_0c_3x3", "features.8.branch2.2.conv"),
    ("InceptionV4/Mixed_5d/Branch_3/Conv2d_0b_1x1", "features.8.branch3.1.conv"),
    # InceptionA (4)
    ("InceptionV4/Mixed_5e/Branch_0/Conv2d_0a_1x1", "features.9.branch0.conv"),
    ("InceptionV4/Mixed_5e/Branch_1/Conv2d_0a_1x1", "features.9.branch1.0.conv"),
    ("InceptionV4/Mixed_5e/Branch_1/Conv2d_0b_3x3", "features.9.branch1.1.conv"),
    ("InceptionV4/Mixed_5e/Branch_2/Conv2d_0a_1x1", "features.9.branch2.0.conv"),
    ("InceptionV4/Mixed_5e/Branch_2/Conv2d_0b_3x3", "features.9.branch2.1.conv"),
    ("InceptionV4/Mixed_5e/Branch_2/Conv2d_0c_3x3", "features.9.branch2.2.conv"),
    ("InceptionV4/Mixed_5e/Branch_3/Conv2d_0b_1x1", "features.9.branch3.1.conv"),
    # ReductionA (Mixed_6a)
    ("InceptionV4/Mixed_6a/Branch_0/Conv2d_1a_3x3", "features.10.branch0.conv"),
    ("InceptionV4/Mixed_6a/Branch_1/Conv2d_0a_1x1", "features.10.branch1.0.conv"),
    ("InceptionV4/Mixed_6a/Branch_1/Conv2d_0b_3x3", "features.10.branch1.1.conv"),
    ("InceptionV4/Mixed_6a/Branch_1/Conv2d_1a_3x3", "features.10.branch1.2.conv"),
    # InceptionB (1)
    ("InceptionV4/Mixed_6b/Branch_0/Conv2d_0a_1x1", "features.11.branch0.conv"),
    ("InceptionV4/Mixed_6b/Branch_1/Conv2d_0a_1x1", "features.11.branch1.0.conv"),
    ("InceptionV4/Mixed_6b/Branch_1/Conv2d_0b_1x7", "features.11.branch1.1.conv"),
    ("InceptionV4/Mixed_6b/Branch_1/Conv2d_0c_7x1", "features.11.branch1.2.conv"),
    ("InceptionV4/Mixed_6b/Branch_2/Conv2d_0a_1x1", "features.11.branch2.0.conv"),
    ("InceptionV4/Mixed_6b/Branch_2/Conv2d_0b_7x1", "features.11.branch2.1.conv"),
    ("InceptionV4/Mixed_6b/Branch_2/Conv2d_0c_1x7", "features.11.branch2.2.conv"),
    ("InceptionV4/Mixed_6b/Branch_2/Conv2d_0d_7x1", "features.11.branch2.3.conv"),
    ("InceptionV4/Mixed_6b/Branch_2/Conv2d_0e_1x7", "features.11.branch2.4.conv"),
    ("InceptionV4/Mixed_6b/Branch_3/Conv2d_0b_1x1", "features.11.branch3.1.conv"),
    # InceptionB (2)
    ("InceptionV4/Mixed_6c/Branch_0/Conv2d_0a_1x1", "features.12.branch0.conv"),
    ("InceptionV4/Mixed_6c/Branch_1/Conv2d_0a_1x1", "features.12.branch1.0.conv"),
    ("InceptionV4/Mixed_6c/Branch_1/Conv2d_0b_1x7", "features.12.branch1.1.conv"),
    ("InceptionV4/Mixed_6c/Branch_1/Conv2d_0c_7x1", "features.12.branch1.2.conv"),
    ("InceptionV4/Mixed_6c/Branch_2/Conv2d_0a_1x1", "features.12.branch2.0.conv"),
    ("InceptionV4/Mixed_6c/Branch_2/Conv2d_0b_7x1", "features.12.branch2.1.conv"),
    ("InceptionV4/Mixed_6c/Branch_2/Conv2d_0c_1x7", "features.12.branch2.2.conv"),
    ("InceptionV4/Mixed_6c/Branch_2/Conv2d_0d_7x1", "features.12.branch2.3.conv"),
    ("InceptionV4/Mixed_6c/Branch_2/Conv2d_0e_1x7", "features.12.branch2.4.conv"),
    ("InceptionV4/Mixed_6c/Branch_3/Conv2d_0b_1x1", "features.12.branch3.1.conv"),
    # InceptionB (3)
    ("InceptionV4/Mixed_6d/Branch_0/Conv2d_0a_1x1", "features.13.branch0.conv"),
    ("InceptionV4/Mixed_6d/Branch_1/Conv2d_0a_1x1", "features.13.branch1.0.conv"),
    ("InceptionV4/Mixed_6d/Branch_1/Conv2d_0b_1x7", "features.13.branch1.1.conv"),
    ("InceptionV4/Mixed_6d/Branch_1/Conv2d_0c_7x1", "features.13.branch1.2.conv"),
    ("InceptionV4/Mixed_6d/Branch_2/Conv2d_0a_1x1", "features.13.branch2.0.conv"),
    ("InceptionV4/Mixed_6d/Branch_2/Conv2d_0b_7x1", "features.13.branch2.1.conv"),
    ("InceptionV4/Mixed_6d/Branch_2/Conv2d_0c_1x7", "features.13.branch2.2.conv"),
    ("InceptionV4/Mixed_6d/Branch_2/Conv2d_0d_7x1", "features.13.branch2.3.conv"),
    ("InceptionV4/Mixed_6d/Branch_2/Conv2d_0e_1x7", "features.13.branch2.4.conv"),
    ("InceptionV4/Mixed_6d/Branch_3/Conv2d_0b_1x1", "features.13.branch3.1.conv"),
    # InceptionB (4)
    ("InceptionV4/Mixed_6e/Branch_0/Conv2d_0a_1x1", "features.14.branch0.conv"),
    ("InceptionV4/Mixed_6e/Branch_1/Conv2d_0a_1x1", "features.14.branch1.0.conv"),
    ("InceptionV4/Mixed_6e/Branch_1/Conv2d_0b_1x7", "features.14.branch1.1.conv"),
    ("InceptionV4/Mixed_6e/Branch_1/Conv2d_0c_7x1", "features.14.branch1.2.conv"),
    ("InceptionV4/Mixed_6e/Branch_2/Conv2d_0a_1x1", "features.14.branch2.0.conv"),
    ("InceptionV4/Mixed_6e/Branch_2/Conv2d_0b_7x1", "features.14.branch2.1.conv"),
    ("InceptionV4/Mixed_6e/Branch_2/Conv2d_0c_1x7", "features.14.branch2.2.conv"),
    ("InceptionV4/Mixed_6e/Branch_2/Conv2d_0d_7x1", "features.14.branch2.3.conv"),
    ("InceptionV4/Mixed_6e/Branch_2/Conv2d_0e_1x7", "features.14.branch2.4.conv"),
    ("InceptionV4/Mixed_6e/Branch_3/Conv2d_0b_1x1", "features.14.branch3.1.conv"),
    # InceptionB (5)
    ("InceptionV4/Mixed_6f/Branch_0/Conv2d_0a_1x1", "features.15.branch0.conv"),
    ("InceptionV4/Mixed_6f/Branch_1/Conv2d_0a_1x1", "features.15.branch1.0.conv"),
    ("InceptionV4/Mixed_6f/Branch_1/Conv2d_0b_1x7", "features.15.branch1.1.conv"),
    ("InceptionV4/Mixed_6f/Branch_1/Conv2d_0c_7x1", "features.15.branch1.2.conv"),
    ("InceptionV4/Mixed_6f/Branch_2/Conv2d_0a_1x1", "features.15.branch2.0.conv"),
    ("InceptionV4/Mixed_6f/Branch_2/Conv2d_0b_7x1", "features.15.branch2.1.conv"),
    ("InceptionV4/Mixed_6f/Branch_2/Conv2d_0c_1x7", "features.15.branch2.2.conv"),
    ("InceptionV4/Mixed_6f/Branch_2/Conv2d_0d_7x1", "features.15.branch2.3.conv"),
    ("InceptionV4/Mixed_6f/Branch_2/Conv2d_0e_1x7", "features.15.branch2.4.conv"),
    ("InceptionV4/Mixed_6f/Branch_3/Conv2d_0b_1x1", "features.15.branch3.1.conv"),
    # InceptionB (6)
    ("InceptionV4/Mixed_6g/Branch_0/Conv2d_0a_1x1", "features.16.branch0.conv"),
    ("InceptionV4/Mixed_6g/Branch_1/Conv2d_0a_1x1", "features.16.branch1.0.conv"),
    ("InceptionV4/Mixed_6g/Branch_1/Conv2d_0b_1x7", "features.16.branch1.1.conv"),
    ("InceptionV4/Mixed_6g/Branch_1/Conv2d_0c_7x1", "features.16.branch1.2.conv"),
    ("InceptionV4/Mixed_6g/Branch_2/Conv2d_0a_1x1", "features.16.branch2.0.conv"),
    ("InceptionV4/Mixed_6g/Branch_2/Conv2d_0b_7x1", "features.16.branch2.1.conv"),
    ("InceptionV4/Mixed_6g/Branch_2/Conv2d_0c_1x7", "features.16.branch2.2.conv"),
    ("InceptionV4/Mixed_6g/Branch_2/Conv2d_0d_7x1", "features.16.branch2.3.conv"),
    ("InceptionV4/Mixed_6g/Branch_2/Conv2d_0e_1x7", "features.16.branch2.4.conv"),
    ("InceptionV4/Mixed_6g/Branch_3/Conv2d_0b_1x1", "features.16.branch3.1.conv"),
    # InceptionB (7)
    ("InceptionV4/Mixed_6h/Branch_0/Conv2d_0a_1x1", "features.17.branch0.conv"),
    ("InceptionV4/Mixed_6h/Branch_1/Conv2d_0a_1x1", "features.17.branch1.0.conv"),
    ("InceptionV4/Mixed_6h/Branch_1/Conv2d_0b_1x7", "features.17.branch1.1.conv"),
    ("InceptionV4/Mixed_6h/Branch_1/Conv2d_0c_7x1", "features.17.branch1.2.conv"),
    ("InceptionV4/Mixed_6h/Branch_2/Conv2d_0a_1x1", "features.17.branch2.0.conv"),
    ("InceptionV4/Mixed_6h/Branch_2/Conv2d_0b_7x1", "features.17.branch2.1.conv"),
    ("InceptionV4/Mixed_6h/Branch_2/Conv2d_0c_1x7", "features.17.branch2.2.conv"),
    ("InceptionV4/Mixed_6h/Branch_2/Conv2d_0d_7x1", "features.17.branch2.3.conv"),
    ("InceptionV4/Mixed_6h/Branch_2/Conv2d_0e_1x7", "features.17.branch2.4.conv"),
    ("InceptionV4/Mixed_6h/Branch_3/Conv2d_0b_1x1", "features.17.branch3.1.conv"),
    # ReductionB (Mixed_7a)
    ("InceptionV4/Mixed_7a/Branch_0/Conv2d_0a_1x1", "features.18.branch0.0.conv"),
    ("InceptionV4/Mixed_7a/Branch_0/Conv2d_1a_3x3", "features.18.branch0.1.conv"),
    ("InceptionV4/Mixed_7a/Branch_1/Conv2d_0a_1x1", "features.18.branch1.0.conv"),
    ("InceptionV4/Mixed_7a/Branch_1/Conv2d_0b_1x7", "features.18.branch1.1.conv"),
    ("InceptionV4/Mixed_7a/Branch_1/Conv2d_0c_7x1", "features.18.branch1.2.conv"),
    ("InceptionV4/Mixed_7a/Branch_1/Conv2d_1a_3x3", "features.18.branch1.3.conv"),
    # InceptionC (1)
    ("InceptionV4/Mixed_7b/Branch_0/Conv2d_0a_1x1", "features.19.branch0.conv"),
    ("InceptionV4/Mixed_7b/Branch_1/Conv2d_0a_1x1", "features.19.branch1_0.conv"),
    ("InceptionV4/Mixed_7b/Branch_1/Conv2d_0b_1x3", "features.19.branch1_1a.conv"),
    ("InceptionV4/Mixed_7b/Branch_1/Conv2d_0c_3x1", "features.19.branch1_1b.conv"),
    ("InceptionV4/Mixed_7b/Branch_2/Conv2d_0a_1x1", "features.19.branch2_0.conv"),
    ("InceptionV4/Mixed_7b/Branch_2/Conv2d_0b_3x1", "features.19.branch2_1.conv"),
    ("InceptionV4/Mixed_7b/Branch_2/Conv2d_0c_1x3", "features.19.branch2_2.conv"),
    ("InceptionV4/Mixed_7b/Branch_2/Conv2d_0d_1x3", "features.19.branch2_3a.conv"),
    ("InceptionV4/Mixed_7b/Branch_2/Conv2d_0e_3x1", "features.19.branch2_3b.conv"),
    ("InceptionV4/Mixed_7b/Branch_3/Conv2d_0b_1x1", "features.19.branch3.1.conv"),
    # InceptionC (2)
    ("InceptionV4/Mixed_7c/Branch_0/Conv2d_0a_1x1", "features.20.branch0.conv"),
    ("InceptionV4/Mixed_7c/Branch_1/Conv2d_0a_1x1", "features.20.branch1_0.conv"),
    ("InceptionV4/Mixed_7c/Branch_1/Conv2d_0b_1x3", "features.20.branch1_1a.conv"),
    ("InceptionV4/Mixed_7c/Branch_1/Conv2d_0c_3x1", "features.20.branch1_1b.conv"),
    ("InceptionV4/Mixed_7c/Branch_2/Conv2d_0a_1x1", "features.20.branch2_0.conv"),
    ("InceptionV4/Mixed_7c/Branch_2/Conv2d_0b_3x1", "features.20.branch2_1.conv"),
    ("InceptionV4/Mixed_7c/Branch_2/Conv2d_0c_1x3", "features.20.branch2_2.conv"),
    ("InceptionV4/Mixed_7c/Branch_2/Conv2d_0d_1x3", "features.20.branch2_3a.conv"),
    ("InceptionV4/Mixed_7c/Branch_2/Conv2d_0e_3x1", "features.20.branch2_3b.conv"),
    ("InceptionV4/Mixed_7c/Branch_3/Conv2d_0b_1x1", "features.20.branch3.1.conv"),
    # InceptionC (3)
    ("InceptionV4/Mixed_7d/Branch_0/Conv2d_0a_1x1", "features.21.branch0.conv"),
    ("InceptionV4/Mixed_7d/Branch_1/Conv2d_0a_1x1", "features.21.branch1_0.conv"),
    ("InceptionV4/Mixed_7d/Branch_1/Conv2d_0b_1x3", "features.21.branch1_1a.conv"),
    ("InceptionV4/Mixed_7d/Branch_1/Conv2d_0c_3x1", "features.21.branch1_1b.conv"),
    ("InceptionV4/Mixed_7d/Branch_2/Conv2d_0a_1x1", "features.21.branch2_0.conv"),
    ("InceptionV4/Mixed_7d/Branch_2/Conv2d_0b_3x1", "features.21.branch2_1.conv"),
    ("InceptionV4/Mixed_7d/Branch_2/Conv2d_0c_1x3", "features.21.branch2_2.conv"),
    ("InceptionV4/Mixed_7d/Branch_2/Conv2d_0d_1x3", "features.21.branch2_3a.conv"),
    ("InceptionV4/Mixed_7d/Branch_2/Conv2d_0e_3x1", "features.21.branch2_3b.conv"),
    ("InceptionV4/Mixed_7d/Branch_3/Conv2d_0b_1x1", "features.21.branch3.1.conv"),
    # Fully-connected layer
    ("InceptionV4/Logits/Logits", "last_linear"),
    # There is also an auxiliary head but the pytorch impl does not include it.
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
        if "/Conv2d" in tf_weights:
            # (1, 3, 512, 256) -> (256, 512, 1, 3)
            tf_weight_array = tf_weight_array.transpose([3, 2, 0, 1])
        elif tf_weights == "InceptionV4/Logits/Logits/weights":
            # (1536, 2) -> (2, 1536)
            tf_weight_array = tf_weight_array.transpose([1, 0])
        new_state_dict[torch_weights] = torch.from_numpy(tf_weight_array)
        new_state_dict[torch_biases] = torch.from_numpy(tf_bias_array)

    if inceptionv4 is None:
        print("Not testing model weights because inceptionv4 module not found.")
    else:
        # Test that conversion was (probably) done correctly.
        true_model = inceptionv4(num_classes=num_classes, pretrained=False)
        if true_model.state_dict().keys() != new_state_dict.keys():
            raise RuntimeError(
                "Something went wrong... converted model keys do not match InceptionV4"
                " keys in PyTorch."
            )
        true_state_dict = true_model.state_dict()
        for true_k, new_k in zip(true_state_dict, new_state_dict):
            true_shape = true_state_dict[true_k].shape
            new_shape = new_state_dict[new_k].shape
            if true_shape != new_shape:
                raise ValueError(
                    f"Shape mismatch in {true_k}: {true_shape} vs {new_shape}"
                )

    torch.save(new_state_dict, output_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", help="Path to the InceptionV4 tensorflow checkpoint.")
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
