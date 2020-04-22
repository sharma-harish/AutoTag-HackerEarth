import pretrainedmodels
import torch.nn as nn

class ResNet34(nn.Module):
    def __init__(self, pretrained):
        super(ResNet34, self).__init__()
        if pretrained:
            self.model = pretrainedmodels.__dict__['resnet34'](pretrained = 'imagenet')
        else:
            self.model = pretrainedmodels.__dict__['resnet34'](pretrained = None)

        self.l0 = nn.Linear(512, 4)
        
    def forward(self, X):
        bs, _, _, _ = X.shape
        X = self.model.features(X)
        X = nn.functional.adaptive_avg_pool2d(X, 1).reshape(bs, -1)
        l0 = self.l0(X)

        return l0