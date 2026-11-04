"""The two models: a CIFAR ResNet-18 with real work per step, and an MLP with almost none."""

from omegaconf import DictConfig
from torch import nn
from torchvision.models import resnet18


def build_model(cfg: DictConfig) -> nn.Module:
    """Build the model named by `cfg.name`.

    Args:
        cfg: The `model` section of the config.

    Returns:
        A module mapping (B, 3, 32, 32) images to (B, num_classes) logits.

    """
    if cfg.name == "resnet18":
        model = resnet18(num_classes=cfg.num_classes)
        # 32x32 inputs: the ImageNet stem (7x7 stride 2, then a max-pool) would shrink the
        # image to 8x8 before the first block.
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        return model
    if cfg.name == "mlp":
        layers: list[nn.Module] = [nn.Flatten()]
        width = 3 * 32 * 32
        for hidden in cfg.mlp_hidden:
            layers += [nn.Linear(width, hidden), nn.ReLU()]
            width = hidden
        layers.append(nn.Linear(width, cfg.num_classes))
        return nn.Sequential(*layers)
    raise ValueError(f"unknown model {cfg.name!r}")
