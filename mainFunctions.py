
import torch
import torch.nn as nn

import random, os, time
import numpy as np
from resnet import ResNet18
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
    assert(args.arch == 'resnet18', "Not Implemented archtecture %s" % args.arch)
    return ResNet18(mnist=False).to(torch.device(args.gpu))

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