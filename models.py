import torch
import torchvision


def resnet34(num_classes: int, state_dict_path=None) -> torchvision.models.ResNet:
    """Create a ResNet34 with a custom number of output classes."""
    model = torchvision.models.resnet34()
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    if state_dict_path is not None:
        print("Loading state dict")
        print(f"  {state_dict_path}")
        state_dict = torch.load(state_dict_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
    return model


# TODO: add inception and vgg models.
