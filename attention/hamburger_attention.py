import torch
from model import SwinUNETR
m = SwinUNETR(in_channels=4, out_channels=3, feature_size=24)
print('model instantiated')
x = torch.randn(1,4,96,96,96)
with torch.no_grad():
     y = m(x)
print('output shape:', y.shape)
print('output min/max:', float(y.min()), float(y.max()))
print('output dtype:', y.dtype)