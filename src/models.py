"""The models, picked by name in the config: our small CNN, a CIFAR ResNet-18, and an MLP."""

from omegaconf import DictConfig
from torch import Tensor, nn
from torchvision.models import resnet18


# The model from notebooks/prototyping.ipynb, unchanged: the model is research, not boilerplate.
def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    """Two 3x3 conv-BN-ReLU layers, then a 2x2 max-pool that halves height and width."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    """(B, 3, 32, 32) images -> (B, num_classes) logits."""

    def __init__(self, num_classes: int = 10, width: int = 64) -> None:
        """Build the three blocks and the classifier head."""
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, width),
            conv_block(width, 2 * width),
            conv_block(2 * width, 4 * width),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(4 * width, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Features, global average pool, then logits."""
        return self.head(self.features(x))


def build_model(cfg: DictConfig) -> nn.Module:
    """Build the model named by `cfg.name`.

    Args:
        cfg: The `model` section of the config.

    Returns:
        A module mapping (B, 3, 32, 32) images to (B, num_classes) logits.

    """
    if cfg.name == "small_cnn":
        return SmallCNN(num_classes=cfg.num_classes, width=cfg.width)
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
