from cnn_classifier.model import CNN_Text
from torch import nn
import torch.nn.functional as F

class bi_CNN_Text(nn.Module):
    def __init__(self, *args, **kwargs):
        super(bi_CNN_Text, self).__init__()
        self.args = args
        self.cnn = CNN_Text(*args, **kwargs)

    def forward(self, x):
        s1, s2 = x
        s1 = self.cnn.confidence(s1)
        s2 = self.cnn.confidence(s2)
        return s1, s2

    def compute_similarity(self, x):
        s1, s2 = self.forward(x)
        y = F.cosine_similarity(s1, s2)
        return y