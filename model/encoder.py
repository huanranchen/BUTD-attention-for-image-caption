import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torchvision import transforms


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.resnet = torchvision.models.resnet101(pretrained=True)
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.resnet.eval()

    def forward(self, x, att_dim=14):
        x = self.transforms(x)

        # (1, C, H, W)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        # print(x.shape)  torch.Size([1, 2048, 16, 12])
        # assert False

        # mean feature
        fc = x.mean(3).mean(2).squeeze()

        # grid feature
        att = F.adaptive_avg_pool2d(x, [att_dim, att_dim]).squeeze()

        return fc, att
