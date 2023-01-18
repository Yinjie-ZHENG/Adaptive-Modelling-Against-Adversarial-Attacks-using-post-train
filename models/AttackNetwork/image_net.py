import torch
import torch.nn as nn
import torchvision.models as models





def image_net():
    #model = models.__dict__['resnet50'](pretrained=False)
    model_ft = models.__dict__['resnet18'](pretrained=False)
    model_ft.avgpool = nn.AdaptiveAvgPool2d(1)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 200)

    return model_ft

def image_net_pretrained():
    #model = models.__dict__['resnet50'](pretrained=False)

    model_ft = models.__dict__['resnet18'](pretrained=True)
    model_ft.avgpool = nn.AdaptiveAvgPool2d(1)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 200)

    return model_ft

def densenet121_pretrained():

    model_ft = models.__dict__['densenet121'](pretrained=True)
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier  = nn.Linear(num_ftrs, 200)

    return model_ft


def resnet50_pretrained():

    model_ft = models.__dict__['resnet50'](pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 200)

    return model_ft
#print("done")