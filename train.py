import torch
from utils.config import *
import time
import numpy as np

global_step = 0
def train(net, train_loader, device, optimizer, criterion, writer=None):
    global global_step
    net.train()
    train_loss = 0.0
    total = 0
    correct = 0
    for i, data in enumerate(train_loader, 0):
        # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
        global_step += 1
        inputs, labels = data[0].to(device), data[1].to(device)

        # 변화도(Gradient) 매개변수를 0으로 만들고
        optimizer.zero_grad()

        # 순전파 + 역전파 + 최적화를 한 후
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # step_lr_scheduler.step()

        # 통계를 출력합니다.
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    acc = 100 * correct / total
    if writer:
        writer.add_scalar("log/train error", 100 - acc, global_step)
        writer.add_scalar("log/train loss", train_loss, global_step)
        
    return train_loss

def getLossPlot(net, test_loader, device, criterion, min_idx, max_idx):
    net.to(device)
    net.eval()
    
    step = 1
    number = 21
    base = -10
    D1 = np.arange(base, base+step*number, step)
    D2 = np.arange(base, base+step*number, step)
    
    Z = np.zeros((len(D1), len(D2)))

    with torch.no_grad():
        ori_wei = net.layer4[1].conv1.weight.data.clone()
        delta = ori_wei.view(-1)[max_idx]*80000
        for x, d1 in enumerate(D1):
            for y, d2 in enumerate(D2):
                net.layer4[1].conv1.weight.data.view(-1)[max_idx] = ori_wei.view(-1)[max_idx] + (d1 * delta)
                net.layer4[1].conv1.weight.data.view(-1)[min_idx] = ori_wei.view(-1)[min_idx] + (d2 * delta)
                test_loss = 0.0
                total = 0
                correct = 0
                for i, data in enumerate(test_loader, 0):
                    inputs, labels = data[0].to(device), data[1].to(device)

                    # 순전파 + 역전파 + 최적화를 한 후
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    # step_lr_scheduler.step()

                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

                acc = 100 * correct / total
                Z[x, y] = test_loss
                
                print(f"{net.layer4[1].conv1.weight.data.view(-1)[max_idx]} {net.layer4[1].conv1.weight.data.view(-1)[min_idx]} {test_loss} {acc} %")
        with open("dumpZ.txt", "w") as f:
            for x, d1 in enumerate(D1):
                for y, d2 in enumerate(D2):
                    f.write(f" {Z[x, y]}")
                f.write('\n')
    # return test_loss
    
def eval(model, test_loader, device, criterion, bestAcc, log=None, writer=None):
    global global_step, PATH
    correct = 0
    total = 0
    total_loss = 0.0
    testNet = model.eval()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            # 신경망에 이미지를 통과시켜 출력을 계산합니다
            st = time.time()
            test_outputs = testNet(images)
            et = time.time()
            # 가장 높은 값(energy)를 갖는 분류(class)를 정답으로 선택하겠습니다
            loss = criterion(test_outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(test_outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    # if acc > bestAcc:
    #     torch.save(model.state_dict(), PATH)
    #     bestAcc = acc
    # if writer:
    #     writer.add_scalar("log/test error", 100-acc, global_step)
    
    log['acc'] = acc
    log['loss'] = total_loss
    
    ##- Test Result
    elapsedTime = '{:0.4f} ms'.format((et - st) * 1000)
    if not loss:
        print(f'[Test ] Accuracy : Loss({total_loss:.3f}): {100 * correct / total:.3f} % - {elapsedTime}')
    return bestAcc

def quantize(model, train_loader, test_loader, device, fbgemm=False):
    model.to(device)
    model.eval()
    total = 0
    modules_to_fuse = [['conv1', 'bn1'],
            ['layer1.0.conv1', 'layer1.0.bn1'],
            ['layer1.0.conv2', 'layer1.0.bn2'],
            ['layer1.1.conv1', 'layer1.1.bn1'],
            ['layer1.1.conv2', 'layer1.1.bn2'],
            ['layer2.0.conv1', 'layer2.0.bn1'],
            ['layer2.0.conv2', 'layer2.0.bn2'],
            ['layer2.0.shortcut.0', 'layer2.0.shortcut.1'],
            ['layer2.1.conv1', 'layer2.1.bn1'],
            ['layer2.1.conv2', 'layer2.1.bn2'],
            ['layer3.0.conv1', 'layer3.0.bn1'],
            ['layer3.0.conv2', 'layer3.0.bn2'],
            ['layer3.0.shortcut.0', 'layer3.0.shortcut.1'],
            ['layer3.1.conv1', 'layer3.1.bn1'],
            ['layer3.1.conv2', 'layer3.1.bn2'],
            ['layer4.0.conv1', 'layer4.0.bn1'],
            ['layer4.0.conv2', 'layer4.0.bn2'],
            ['layer4.0.shortcut.0', 'layer4.0.shortcut.1'],
            ['layer4.1.conv1', 'layer4.1.bn1'],
            ['layer4.1.conv2', 'layer4.1.bn2']]
    if fuseModel:
        model = torch.quantization.fuse_modules(model, modules_to_fuse)
    if fbgemm:
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    else:
        model.qconfig = torch.quantization.default_qconfig
    torch.quantization.prepare(model, inplace=True)
    model.eval()
    with torch.no_grad():
        for data, target in train_loader:
            model(data.to(device))
    torch.quantization.convert(model, inplace=True)
    
    correct = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            st = time.time()
            test_outputs = model(images)
            et = time.time()
            _, predicted = torch.max(test_outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f'Quant Accuracy: {acc:.3f} %')
    print('Evaluation Elapsed time = {:0.4f} milliseconds'.format((et - st) * 1000))    
    

            
