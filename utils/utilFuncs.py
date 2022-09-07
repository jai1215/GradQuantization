import time
import torch
import re
import time

def printStatus(model, optimizer):
    # 모델의 state_dict 출력
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size(), model.state_dict()[param_tensor].dtype)
        
    # # 옵티마이저의 state_dict 출력
    # print("Optimizer's state_dict:")
    # for var_name in optimizer.state_dict():
    #     print(var_name, "\t", optimizer.state_dict()[var_name])

def product(lst):
    ret = 1
    for l in lst:
        ret *= l
    return ret

def visualise(x, axs):
    x = x.view(-1).cpu().numpy()
    axs.hist(x) 
    
# Entry function to get stats of all functions.
def gatherStats(model, test_loader):
    device = 'cuda:0'
    
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            model.updateRange(data)
    model.rangeUpdate(pprint=False)

def dumpGrad(model, filename):
    grads = {}
    for name, param in model.named_parameters():
        if re.search(r'conv.\.weight', name):
            grads[name] = param.grad

    torch.save(grads, filename)

def logging_time(original_fn):
    def wrapper_fn(*args, **kwargs):
        start_time = time.time()
        result = original_fn(*args, **kwargs)
        end_time = time.time()
        print("Funtion Running[{}]: {} sec".format(original_fn.__name__, end_time-start_time))
        return result
    return wrapper_fn
