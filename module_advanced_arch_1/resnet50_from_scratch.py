#!/usr/bin/env python3
"""
Minimal ResNet-50 implementation in PyTorch.

Run:
    python resnet50_from_scratch.py

Optional:
    python resnet50_from_scratch.py --batch-size 4 --num-classes 10 --show-shapes
"""

import argparse
import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, bottleneck_channels, stride=1, downsample=None):
        super().__init__()
        out_channels = bottleneck_channels * self.expansion

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=bottleneck_channels,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)

        self.conv2 = nn.Conv2d(
            in_channels=bottleneck_channels,
            out_channels=bottleneck_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)

        self.conv3 = nn.Conv2d(
            in_channels=bottleneck_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, show_shapes=False):
        super().__init__()
        self.in_channels = 64
        self.show_shapes = show_shapes

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, bottleneck_channels=64, blocks=layers[0], stride=1)
        self.layer2 = self._make_layer(block, bottleneck_channels=128, blocks=layers[1], stride=2)
        self.layer3 = self._make_layer(block, bottleneck_channels=256, blocks=layers[2], stride=2)
        self.layer4 = self._make_layer(block, bottleneck_channels=512, blocks=layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, bottleneck_channels, blocks, stride):
        out_channels = bottleneck_channels * block.expansion

        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

        layers = [
            block(
                in_channels=self.in_channels,
                bottleneck_channels=bottleneck_channels,
                stride=stride,
                downsample=downsample,
            )
        ]

        self.in_channels = out_channels

        for _ in range(1, blocks):
            layers.append(
                block(
                    in_channels=self.in_channels,
                    bottleneck_channels=bottleneck_channels,
                    stride=1,
                    downsample=None,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.show_shapes:
            print("Input:         ", x.shape)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.show_shapes:
            print("After conv1:   ", x.shape)

        x = self.maxpool(x)
        if self.show_shapes:
            print("After maxpool: ", x.shape)

        x = self.layer1(x)
        if self.show_shapes:
            print("After layer1:  ", x.shape)

        x = self.layer2(x)
        if self.show_shapes:
            print("After layer2:  ", x.shape)

        x = self.layer3(x)
        if self.show_shapes:
            print("After layer3:  ", x.shape)

        x = self.layer4(x)
        if self.show_shapes:
            print("After layer4:  ", x.shape)

        x = self.avgpool(x)
        if self.show_shapes:
            print("After avgpool: ", x.shape)

        x = torch.flatten(x, 1)
        if self.show_shapes:
            print("After flatten: ", x.shape)

        x = self.fc(x)
        if self.show_shapes:
            print("After fc:      ", x.shape)

        return x


def resnet50(num_classes=1000, show_shapes=False):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, show_shapes=show_shapes)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parse_args():
    parser = argparse.ArgumentParser(description="Run a minimal ResNet-50 example.")
    parser.add_argument("--batch-size", type=int, default=1, help="Input batch size.")
    parser.add_argument("--num-classes", type=int, default=1000, help="Number of output classes.")
    parser.add_argument("--image-size", type=int, default=224, help="Square input image size.")
    parser.add_argument("--show-shapes", action="store_true", help="Print tensor shapes during the forward pass.")
    return parser.parse_args()


def main():
    args = parse_args()

    model = resnet50(num_classes=args.num_classes, show_shapes=args.show_shapes)
    model.eval()

    x = torch.randn(args.batch_size, 3, args.image_size, args.image_size)

    with torch.no_grad():
        y = model(x)

    print("\nModel created successfully.")
    print(f"Trainable parameters: {count_parameters(model):,}")
    print(f"Output shape: {tuple(y.shape)}")
    print(f"Output Logits: {y}")


if __name__ == "__main__":
    main()


