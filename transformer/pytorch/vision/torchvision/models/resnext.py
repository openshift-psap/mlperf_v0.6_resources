import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNeXt', 'resnext50', 'resnext101',
           'resnext152']


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ResNeXtBottleneckC(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, cardinality=32, baseWidth=4):
        super(ResNeXtBottleneckC, self).__init__()

        width = math.floor(planes/64 * cardinality * baseWidth)

        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=cardinality)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNeXt(nn.Module):

    def __init__(self, block, layers, num_classes=1000, cardinality=32, baseWidth=4, shortcut='C'):
        self.inplanes = 64
        self.cardinality = cardinality
        self.baseWidth = baseWidth
        self.shortcut = shortcut
        super(ResNeXt, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64,  layers[0])
        self.layer2 = self._make_layer(block, 128,  layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256,  layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        reshape = stride != 1 or self.inplanes != planes * block.expansion
        useConv = (self.shortcut == 'C') or (self.shortcut == 'B' and reshape) 

        if useConv:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        elif reshape:
            downsample = nn.AvgPool2d(3, stride=stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.cardinality, self.baseWidth))
        self.inplanes = planes * block.expansion

        if self.shortcut == 'C':
            shortcut = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
        else:
            shortcut = None
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, shortcut, self.cardinality, self.baseWidth))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnext50(cardinality=32, baseWidth=4, shortcut='C', **kwargs):
    """Constructs a ResNeXt-50 model.

    Args:
        cardinality (int): Cardinality of the aggregated transform
        baseWidth (int): Base width of the grouped convolution
        shortcut ('A'|'B'|'C'): 'B' use 1x1 conv to downsample, 'C' use 1x1 conv on every residual connection
    """
    model = ResNeXt(ResNeXtBottleneckC, [3, 4, 6, 3], cardinality=cardinality, baseWidth=baseWidth, shortcut=shortcut, **kwargs)
    return model


def resnext101(cardinality=32, baseWidth=4, shortcut='C', **kwargs):
    """Constructs a ResNeXt-101 model.

    Args:
        cardinality (int): Cardinality of the aggregated transform
        baseWidth (int): Base width of the grouped convolution
        shortcut ('A'|'B'|'C'): 'B' use 1x1 conv to downsample, 'C' use 1x1 conv on every residual connection
    """
    model = ResNeXt(ResNeXtBottleneckC, [3, 4, 23, 3], cardinality=cardinality, baseWidth=baseWidth, shortcut=shortcut, **kwargs)
    return model


def resnext152(cardinality=32, baseWidth=4, shortcut='C', **kwargs):
    """Constructs a ResNeXt-152 model.

    Args:
        cardinality (int): Cardinality of the aggregated transform
        baseWidth (int): Base width of the grouped convolution
        shortcut ('A'|'B'|'C'): 'B' use 1x1 conv to downsample, 'C' use 1x1 conv on every residual connection
    """
    model = ResNeXt(ResNeXtBottleneckC, [3, 8, 36, 3], cardinality=cardinality, baseWidth=baseWidth, shortcut=shortcut, **kwargs)
    return model

