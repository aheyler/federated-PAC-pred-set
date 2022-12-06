from model.util import *
from model.resnet import ResNet18, ResNet50, ResNet101, ResNet152, ResNetFeat
from model.pred_set import PredSet, PredSetCls, PredSetReg
from model.pred_set_federated import PredSetFederated, PredSetFederatedCls, PredSetRegFederated
from model.split_cp import SplitCPCls, SplitCPReg, WeightedSplitCPCls

from model.fnn import Linear, SmallFNN, MidFNN, BigFNN
from model.fnn_reg import LinearReg, SmallFNNReg, MidFNNReg, BigFNNReg

from model.resnets_no_bn import conv3x3, conv1x1, BasicBlock, Bottleneck, ResNet, ResNet18_Weights, \
    ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights, ResNeXt50_32X4D_Weights, \
        ResNeXt101_32X8D_Weights, ResNeXt101_64X4D_Weights, Wide_ResNet50_2_Weights, Wide_ResNet101_2_Weights