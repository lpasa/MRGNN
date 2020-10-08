from torch.nn import Linear
from torch.nn import functional as F
from torch.nn.utils import spectral_norm
from torch.nn import init
import math
from torch.nn.init import xavier_uniform_


class Linear_masked_weight(Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(Linear_masked_weight, self).__init__(in_features, out_features, bias)
        self.weight.requires_grad =False
        self.bias.requires_grad = False

    def forward(self, input, mask):
        maskedW=self.weight*mask
        return F.linear(input, maskedW, self.bias)
