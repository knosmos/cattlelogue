import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    """
    U-Net architecture for image segmentation tasks.
    Four encoder blocks followed by a bottleneck and four decoder blocks.
    Each encoder block halves the spatial dimensions while doubling the number of channels.
    Aggressive dropout by default to prevent overfitting.
    """

    def __init__(self, in_channels=52, out_channels=1, dropout_rate=0.3):
        super(UNet, self).__init__()
        self.dropout_rate = dropout_rate

        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        self.bottleneck = self.conv_block(512, 1024)

        # Decoder block shapes have this structure
        # since we do concatenation (skip connections) with the encoder layers
        self.decoder4 = self.upconv_block(1024, 512)
        self.decoder3 = self.upconv_block(1024, 256)
        self.decoder2 = self.upconv_block(512, 128)
        self.decoder1 = self.upconv_block(256, 64)

        self.final_conv = nn.Conv2d(128, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_rate),
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_rate),
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, kernel_size=2))
        enc3 = self.encoder3(F.max_pool2d(enc2, kernel_size=2))
        enc4 = self.encoder4(F.max_pool2d(enc3, kernel_size=2))

        bottleneck = self.bottleneck(F.max_pool2d(enc4, kernel_size=2))
        # print("shape after bottleneck:", bottleneck.shape)

        dec4 = self.decoder4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)  # Skip connection
        # print("shape after decoder4:", dec4.shape)
        dec3 = self.decoder3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)  # Skip connection
        # print("shape after decoder3:", dec3.shape)
        dec2 = self.decoder2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)  # Skip connection
        # print("shape after decoder2:", dec2.shape)
        dec1 = self.decoder1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)  # Skip connection
        out = self.final_conv(dec1)
        # print("shape after final conv:", out.shape)

        return out
