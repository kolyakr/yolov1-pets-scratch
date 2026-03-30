from torch import nn

class DetectionHead(nn.Module):
    def __init__(self, in_channels, shrink_channels, expand_channels, S, B, num_classes):
        super().__init__()

        self.S = S
        self.B = B
        self.num_classes = num_classes

        self.shrink_conv1x1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=shrink_channels,
            kernel_size=(1, 1),
            stride=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(num_features=shrink_channels)

        self.conv3x3 = nn.Conv2d(
            in_channels=shrink_channels,
            out_channels=shrink_channels,
            kernel_size=(3, 3),
            stride=2,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(num_features=shrink_channels)

        self.expand_conv1x1 = nn.Conv2d(
            in_channels=shrink_channels,
            out_channels=expand_channels,
            kernel_size=(1, 1),
            stride=1,
            bias=False
        )

        self.bn3 = nn.BatchNorm2d(num_features=expand_channels)

        self.leaky_relu = nn.LeakyReLU(0.1)

        self.flat = nn.Flatten()

        self.dense1 = nn.Linear(
            in_features=expand_channels * S * S,
            out_features=4096
        )

        self.dropout = nn.Dropout(0.5)

        self.dense2 = nn.Linear(
            in_features=4096, 
            out_features= S * S * (num_classes + B * 5)
        )

    def forward(self, X):

        X = self.leaky_relu(self.bn1(self.shrink_conv1x1(X)))
        X = self.leaky_relu(self.bn2(self.conv3x3(X)))
        X = self.leaky_relu(self.bn3(self.expand_conv1x1(X)))

        X = self.flat(X)

        X = self.leaky_relu(self.dense1(X))
        X = self.dropout(X)
        out = self.dense2(X)

        return out.view(out.shape[0], self.S, self.S, -1)