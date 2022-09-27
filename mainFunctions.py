
import torch
import torch.nn as nn

import random, os, time
import numpy as np
# from resnet import ResNet18, ResNet34, ResNet50
from models.resnet import resnet18, resnet34, resnet50
from models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from models.mobilenetv2 import mobilenet_v2
from types import SimpleNamespace
import torch.optim as optim
from torch.optim import lr_scheduler

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def getToday():
    now = time.localtime()
    ret = f"{(now.tm_year%100):02d}"
    ret += f"_{now.tm_mon:02d}_{now.tm_mday:02d}_{now.tm_hour:02d}:{now.tm_min:02d}:{now.tm_sec:02d}"
    return ret

def seed_all(seed=2022):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def getModel(args):
    # assert(args.arch in ['resnet18', 'resnet34', 'resnet50'], "Not Implemented archtecture %s" % args.arch)
    model = None
    if args.arch == 'resnet18':
        model =  resnet18(pretrained=False, wpath=args.load_param_path, device=args.gpu).to(torch.device(args.gpu))
    elif args.arch == 'resnet34':
        model =  resnet34(pretrained=False, wpath=args.load_param_path, device=args.gpu).to(torch.device(args.gpu))
    elif args.arch == 'resnet50':
        model =  resnet50(pretrained=False, wpath=args.load_param_path, device=args.gpu).to(torch.device(args.gpu))
    elif args.arch == 'vgg11_bn':
        model =  vgg11_bn(pretrained=False, device=args.gpu).to(torch.device(args.gpu))
    elif args.arch == 'vgg13_bn':
        model =  vgg13_bn(pretrained=False, device=args.gpu).to(torch.device(args.gpu))
    elif args.arch == 'vgg16_bn':
        model =  vgg16_bn(pretrained=False, device=args.gpu).to(torch.device(args.gpu))
    elif args.arch == 'vgg19_bn':
        model =  vgg19_bn(pretrained=False, device=args.gpu).to(torch.device(args.gpu))
    elif args.arch == 'mobilenet_v2':
        model =  mobilenet_v2(pretrained=False, device=args.gpu).to(torch.device(args.gpu))
    
    if args.load_param:
        # print("Loading model from: ", args.load_param_path)
        state_dict = torch.load(args.load_param_path, map_location=args.gpu)
        model.load_state_dict(state_dict)
        
    return model


def getTrainArgs(args, model):
    ret = SimpleNamespace()
    ret.criterion = nn.CrossEntropyLoss()
    
    ## -- Select Optimizer
    # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    ret.optimizer = optim.Adam(model.parameters(), lr=0.001)
    decay_epoch = [16000, 48000]
    ret.step_lr_scheduler = lr_scheduler.MultiStepLR(ret.optimizer, 
                                    milestones=decay_epoch, gamma=0.1)
    return ret

# -- Summary Option
# summary(model, (200, 3, 32, 32))