from torch import nn
import torch


class ThreeConvModel(nn.Module):
    def __init__(
        self,
        n_conv_blocks
    ):
        self.conv_blocks = []
        for i in range(n_conv_blocks):
            self.conv_blocks.append(nn.Conv3d(1, 1, (1,3,3), stride=1, padding=(0,1,1)))

    def forward(self, x: torch.tensor):
        # Encoder pathway
        for conv in self.conv_blocks:
            x = conv(x)
        return x



'''
class TwoConvModel(nn.Module):
    def __init__(
        self,
        n_conv_blocks
    ):
        self.conv_blocks = []
        for i in range(n_conv_blocks):
            self.conv_blocks.append(nn.Conv2d(1, 1, (3,3,3), stride=1, padding=(0,1,1)))

    def forward(self, x: torch.tensor):
        # Encoder pathway
        for conv in self.conv_blocks:
            x = conv(x)
        return x
'''

if __name__ == "__main__":
    cmodel = ThreeConvModel(n_conv_blocks=3)
    print(len(cmodel.conv_blocks))
    print(cmodel.conv_blocks[0].weight.shape)
    image = torch.rand(1, 1, 3, 896, 1344)
    output = cmodel.forward(image)
    print(output.shape)


    