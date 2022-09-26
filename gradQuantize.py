import re
import torch
from torch.autograd import grad
from utils.utilFuncs import *
from tqdm import tqdm
from mainFunctions import getModel
from train import eval
import os

@logging_time
def generateGrad(model, train_loader, device, criterion, args, optimizer):
    model.train()
    
    ## -- Make dump path
    if not os.path.isdir(args.dump_path):
        os.makedirs(args.dump_path)
        print(f"Making dump path : {args.dump_path}")
    ## -- Save Weights
    for name, param in model.named_parameters():
        if not re.search(r'conv.\.weight', name):
            continue
        torch.save(param, f'{args.dump_path}/{args.TEST}_{name}_weight.pth')
    
    for i, data in enumerate(train_loader):
        ## -- Remove Gradient from model
        optimizer.zero_grad()
        
        if(i >= args.grad_epoch):
            break
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        grads = {}
        for name, param in model.named_parameters():
            if not re.search(r'conv.\.weight', name):
                continue
            
            ## -- Running Grad
            grads[name] = grad(loss, param, create_graph=args.run_hess, retain_graph=True)[0]
            torch.save(grads[name], f'{args.dump_path}/{args.TEST}_{name}_grad_{i:03d}.pth')
            
            ## -- Running Hessian
            if args.run_hess:
                generateHess(grads, name, param, i, args)

def generateHess(grads, name, weight, idx, args):
    pFlatten = grads[name].view(-1)
    weightShape = grads[name].shape
    hessShape = weightShape + weightShape[1:]
    subHessian = torch.zeros(hessShape)

    subSize = (product(weightShape[1:4]), product(weightShape[2:4]), weightShape[3], 1)
    for j, param2 in enumerate(tqdm(pFlatten, desc=name)):
        def idxOf(idx, subSize):
            ret = [j // subSize[0]]
            remain = j % subSize[0]
            ret += [remain // subSize[1]]
            remain = remain % subSize[1]
            ret += [remain // subSize[2]]
            remain = remain % subSize[2]
            ret += [remain]
            return ret
        n, c, fx, fy = idxOf(j, subSize)
        subHessian[n, c, fx, fy] = grad(param2, weight, retain_graph=True)[0][n]
    torch.save(subHessian, f'{args.dump_path}/{args.TEST}_{name}_hess_{idx:03d}.pth')


def testKLQuant(model, bits=8, clipping=1.0, symmetric=False):
    idx = 0
    for name, param in model.named_parameters():
        if re.search(r'conv.\.weight', name):
            param.data = weightKLQuant(name, param, bits)
        idx += 1
    return model

def weightKLQuant(name, param, bits):
    paramOri   = param.clone().detach()
    paramAbs = torch.abs(paramOri).cpu()
    
    minbins = 1024
    maxbins = max(2048, 2**bits*32)
    hist = torch.histogram(paramAbs, bins=maxbins, range=(0., torch.max(paramAbs)))
    edges = hist.bin_edges
    hist = hist.hist
    
    bins = 2**bits
    minBinLen = max(1, minbins // bins)
    maxBinLen = maxbins // bins
    
    #Calculate the KL divergence
    kldivs = []
    for binLen in range(minBinLen, maxBinLen+1):
        kldiv = NvidiaKLDiv(hist, binLen, bins)
        kldivs.append((kldiv, edges[binLen*bins]))

    minKlDiv = 1e10
    minEdge = 0
    for kldiv, edge in kldivs:
        if kldiv < minKlDiv:
            minKlDiv = kldiv
            minEdge = edge
    return quant(param, bits, -minEdge, minEdge)
    
def NvidiaKLDiv(hist, binLen, bins):
    hLen = binLen*bins
    hTotalSum = torch.sum(hist)
    hist[hLen-1] = torch.sum(hist[hLen-1:]).item()
    # hist[hLen:] = 0
    histOri   = hist.clone().detach()
    histQuant = hist
    for b in range(bins):
        histQuant[binLen*b:binLen*(b+1)] = NvidiaKLDivSub(histQuant[binLen*b:binLen*(b+1)], binLen)
    histOri   = histOri / hTotalSum
    histQuant = histQuant / hTotalSum
    return klDiv(histOri[:hLen], histQuant[:hLen])

def NvidiaKLDivSub(hist, hlen):
    hsum = torch.sum(hist) / torch.count_nonzero(hist)
    for i in range(hlen):
        if hist[i] != 0:
            hist[i] = hsum
    return hist

def pushLog(log, key, dat):
    if key in log:
        log[key] += dat
    else:
        log[key] = dat

HessDumped = [
    'conv1.weight',
    'layer1.0.conv1.weight',
    'layer1.0.conv2.weight',
    'layer1.1.conv1.weight',
    'layer1.1.conv2.weight',
    'layer2.0.conv1.weight',
    'layer2.0.conv2.weight',
    'layer2.1.conv1.weight',
    'layer2.1.conv2.weight',
    'layer3.0.conv1.weight',
]

def quantEval(device, model, bits, clipping, log, args, symmetric=False, useHess=True):
    klDivSum = {'run': True, 'sum': 0, 'count': 0}
    
    for name, param in model.named_parameters():
        if re.search(r'conv.\.weight', name):
            paramBefore = param.clone().detach()
            paramGrad   = torch.load(f'{args.dump_path}/{args.TEST}_{name}_grad_000.pth', map_location=device)
            paramHess   = None
            if (name in HessDumped) and useHess: # Load When Hessian Data exists
                paramHess = torch.load(f'{args.dump_path}/{args.TEST}_{name}_hess_000.pth', map_location=device)
            
            ## -- Running Quantization
            param.data = weightQuantize(name, param, bits, clipping, symmetric, klDivSum)
            
            ## -- Calculating emulated Losses
            # pushLog(log, 'mseSumP0.5', meanSquareError(paramBefore, param, power=0.5))
            pushLog(log, 'mseSumP1.0', meanSquareError(paramBefore, param, power=1.0))
            # pushLog(log, 'mseSumP1.5', meanSquareError(paramBefore, param, power=1.5))
            pushLog(log, 'mseSumP2.0', meanSquareError(paramBefore, param, power=2.0))
            # mseSum[4] += meanSquareError(paramBefore, param, power=1, torchAbs=False)
            pushLog(log, 'gradP1.0', meanSquareError(paramBefore, param, power=1.0, weight=paramGrad))
            pushLog(log, 'gradP1.2', meanSquareError(paramBefore, param, power=1.2, weight=paramGrad))
            pushLog(log, 'gradP1.4', meanSquareError(paramBefore, param, power=1.4, weight=paramGrad))
            pushLog(log, 'gradP1.5', meanSquareError(paramBefore, param, power=1.5, weight=paramGrad))
            pushLog(log, 'gradP1.7', meanSquareError(paramBefore, param, power=1.7, weight=paramGrad))
            pushLog(log, 'gradP2.0', meanSquareError(paramBefore, param, power=2.0, weight=paramGrad))
            pushLog(log, 'gradP2.5', meanSquareError(paramBefore, param, power=2.5, weight=paramGrad))
            pushLog(log, 'gradP3.0', meanSquareError(paramBefore, param, power=3.0, weight=paramGrad))
            
            ## -- nSamples
            lossGradP10 = log['gradP1.0']
            lossGradP15 = log['gradP1.5']
            lossGradP20 = log['gradP2.0']
            for nSample in range(1, 16):
                paramGradSub = torch.load(f'{args.dump_path}/{args.TEST}_{name}_grad_{nSample:03d}.pth', map_location=device)
                lossGradP10Sub = meanSquareError(paramBefore, param, power=1.0, weight=paramGradSub)
                lossGradP15Sub = meanSquareError(paramBefore, param, power=1.5, weight=paramGradSub)
                lossGradP20Sub = meanSquareError(paramBefore, param, power=2.0, weight=paramGradSub)
                lossGradP10 += lossGradP10Sub
                lossGradP15 += lossGradP15Sub
                lossGradP20 += lossGradP20Sub
                pushLog(log, f'gradP1.0_{nSample}', lossGradP10)
                pushLog(log, f'gradP1.5_{nSample}', lossGradP15)
                pushLog(log, f'gradP2.0_{nSample}', lossGradP20)
            
            if not useHess:
                continue
            
            pushLog(log, 'hessP1.0', meanSquareError(paramBefore, param, power=1.0, weight=paramGrad, hess=paramHess))
            pushLog(log, 'hessP1.5', meanSquareError(paramBefore, param, power=1.6, weight=paramGrad, hess=paramHess))
            pushLog(log, 'hessP2.0', meanSquareError(paramBefore, param, power=2.0, weight=paramGrad, hess=paramHess))
                        
            pushLog(log, 'L3Loss2.0', meanSquareError(paramBefore, param, power=1.0, weight=paramGrad, hess=paramHess, L3Loss=2.0))
            pushLog(log, 'L3Loss2.5', meanSquareError(paramBefore, param, power=1.0, weight=paramGrad, hess=paramHess, L3Loss=2.5))
            pushLog(log, 'L3Loss3.0', meanSquareError(paramBefore, param, power=1.0, weight=paramGrad, hess=paramHess, L3Loss=3.0))
            pushLog(log, 'L3Loss3.5', meanSquareError(paramBefore, param, power=1.0, weight=paramGrad, hess=paramHess, L3Loss=3.5))
            # pushLog(log, 'noAbsP1.0', meanSquareError(paramBefore, param, power=1.0, weight=paramGrad, torchAbs=False))
            # pushLog(log, 'noAbsP2.0', meanSquareError(paramBefore, param, power=2.0, weight=paramGrad, torchAbs=False))
            # mseGradSum[4] += meanSquareError(paramBefore, param, weight=grad[name], power=1, torchAbs=False)
            # pushLog(log, 'taylor2', talorError(paramBefore, param, paramGrad, 2))
            # pushLog(log, 'taylor3', talorError(paramBefore, param, paramGrad, 3))
            
            # pushLog(log, 'roundP0.5', meanSquareError(paramBefore, param, power=0.5, weight=paramGrad, roundError=True, clipping=clipping))
            # pushLog(log, 'roundP1.0', meanSquareError(paramBefore, param, power=1.0, weight=paramGrad, roundError=True, clipping=clipping))
            # pushLog(log, 'roundP1.5', meanSquareError(paramBefore, param, power=1.5, weight=paramGrad, roundError=True, clipping=clipping))
            # pushLog(log, 'roundP2.0', meanSquareError(paramBefore, param, power=2.0, weight=paramGrad, roundError=True, clipping=clipping))
            # pushLog(log, 'mseSumP0.5', meanSquareError(paramBefore, param, power=0.5))
            
    if klDivSum['run']:
        log['klDiv'] = klDivSum['sum']
    return model

def weightQuantize(name, param, bits=8, clipping=1.0, symmetric=False, klDivSum={}, quantDict={}):
    paramBefore = param.clone().detach()
    param = param.data
    minVal = param.min() * clipping
    maxVal = param.max() * clipping
    if symmetric:
        minValAbs = torch.abs(minVal)
        maxValAbs = torch.abs(maxVal)
        maxTh = torch.max(minValAbs, maxValAbs)
        minVal = -maxTh
        maxVal = maxTh
    quantDict['min'] = minVal
    quantDict['max'] = maxVal
    
    paramAfter = quant(param, bits, minVal, maxVal)
    if 'run' in klDivSum:
        if klDivSum['run']:
            klDivSum['sum'] += klDivParam(paramBefore, paramAfter, bits, maxVal)
            klDivSum['count'] += 1
    return paramAfter

def quant(param, bits, minVal, maxVal):
    param = torch.max(torch.min(param, maxVal), minVal)
    
    scale = (maxVal - minVal) / (2 ** bits - 1)
    
    param = (param - minVal) / scale
    param = torch.round(param)
    param = param * scale + minVal
    return param

def klDivParam(p, q, bits, maxVal):
    maxVal = maxVal.item()
    p = p.flatten().cpu()
    q = q.flatten().cpu()
    p = torch.clamp(p, min=-maxVal, max=maxVal)
    q = torch.clamp(q, min=-maxVal, max=maxVal)
    bins = 2**bits
    histP = torch.histogram(p, bins=bins*32, range=(-maxVal, maxVal))
    histQ = torch.histogram(q, bins=bins, range=(-maxVal, maxVal))
    
    hTotalSum = torch.sum(histP.hist)
    
    histP = histP.hist / hTotalSum
    histQ = histQ.hist / hTotalSum
    histQ = histQ.unsqueeze(1).repeat(1, 32) / 32
    histQ = histQ.flatten()
    return klDiv(histP, histQ).data.item()

def klDiv(p, q):
    assert p.shape == q.shape
    p = torch.abs(p)
    q = torch.abs(q)
    p = p.clamp(min=1e-12)
    q = q.clamp(min=1e-12)
    return torch.sum(p * torch.log(p / q))

def meanSquareError(p, q, weight=None, power=2, torchAbs=True, roundError=False, hess=None, clipping=1.0, L3Loss=0, channel_wise=False):
    delta = torch.abs(p - q) if torchAbs else (p - q)
    if weight is not None:
        weight = torch.abs(weight) if torchAbs else weight
        
    if roundError:
        rTensor = roundTensor(p, clipping)
        delta = rTensor * delta # remove delta when clipping out Area
        
    #Flatten
    n, c, fx, fy = 1, 1, 1, 1
    if channel_wise:
        c, fx, fy = delta.shape
    else:
        n, c, fx, fy = delta.shape
        
    delta   = torch.reshape(delta, (n*c*fx*fy, 1))
    delta_n = torch.reshape(delta, (n, c*fx*fy, 1))
    if weight is not None:
        weight = torch.reshape(weight, (n*c*fx*fy, 1))
    if hess is not None:
        hess = torch.reshape(hess, (n, c*fx*fy, c*fx*fy))
        
    
    #Power
    deltaP   = delta   ** power
    sign = torch.sign(delta_n)
    delta_n = (sign*delta_n) ** power
    delta_n = sign * delta_n
    
    hess_result = []
    
    #Weighting(Grad & Hess)
    ret = 0
    if weight is None:
        ret = torch.sum(deltaP)
    else:
        ret = torch.matmul(deltaP.T, weight)
        
        if hess is not None:
            for i in range(n):
                hess1 = torch.matmul(delta_n[i].T, hess[i])
                hess2 = torch.matmul(hess1, delta_n[i])
                hess_result.append(hess2)
                ret += hess2
    # L3 Loss
    if L3Loss > 0:
        l3 = torch.matmul((delta ** L3Loss).T, weight)/10
        ret += l3
    # devide with number of parameters
    ret = ret / p.numel()
        
    return ret.data.item()

    
def talorError(p, q, deriv, power, normalize=False):
    factorial = [1]
    for pow in range(2, power+1):
        factorial += [factorial[-1] * pow]
    delta = torch.abs(p - q)
    deriv = torch.abs(deriv)
    
    if normalize:
        dmin = torch.min(delta)
        dmax = torch.max(delta)
        delta = (delta - dmin) / (dmax - dmin + 1e-12)
        deriv = (deriv - torch.min(deriv)) / (torch.max(deriv) - torch.min(deriv) + 1e-12)
    tErr = torch.sum(delta * torch.abs(deriv))
    
    tErrSum = tErr
    for pow in range(2, power+1):
        tErrSum += ((tErr**pow) / factorial[pow-1])
    
    # devide with number of parameters
    tErrSum = tErrSum / p.numel()
    return tErrSum

def roundTensor(paramBefore, clipping):
    minVal = paramBefore.min() * clipping
    maxVal = paramBefore.max() * clipping
    maxTh = torch.max(torch.abs(minVal), torch.abs(maxVal))
    ret = (paramBefore <= maxTh) & (paramBefore >= -maxTh)
    return ret.type(torch.uint8)

def quantLayer_with_MSE(device, bit, args, name, param, power=1.0, grad=None, hess=None, channel_wise=False):
    paramBefore = param.clone().detach()
    paramAfter  = param.clone().detach()

    #Quantize resolution
    clipStart = args.quant_clip_start
    clipEnd   = args.quant_clip_end
    quant_resolution = args.quant_resolution
    clipSize  = clipEnd - clipStart
    clipStep  = clipSize / quant_resolution
    
    #Find Minimum point
    minClipping = 1.0
    minMSE = 1e+100
    for clp in range(1, quant_resolution+1):
        clipping = clipStart + clipStep*clp
        paramAfter.data = weightQuantize(name, param, bit, clipping)
        curMSE = meanSquareError(paramBefore, paramAfter, power=power, weight=grad, hess=hess, channel_wise=channel_wise)
        if curMSE < minMSE: #update minimum point
            minClipping = clipping
            minMSE = curMSE
    
    return weightQuantize(name, param, bit, minClipping)
        
def quantModel_with_MSE(device, bit, args, test_loader, train_args, power=1.0, useGrad=False, useHess=False):
    quant_net = getModel(args)
    quant_net.load_state_dict(torch.load(args.load_param_path, map_location=device))
    quant_net.to(torch.device(device))
    for name, param in quant_net.named_parameters():
        if re.search(r'conv.\.weight', name):
            paramGrad   = None
            paramHess   = None
            if useGrad:
                paramGrad   = torch.load(f'{args.dump_path}/{args.TEST}_{name}_grad_000.pth', map_location=device)
            if (name in HessDumped) and args.quant_use_hess: # Load When Hessian Data exists
                paramHess = torch.load(f'{args.dump_path}/{args.TEST}_{name}_hess_000.pth', map_location=device)
            
            if args.run_channelWise:
                #for each output channel
                n_channel = param.data.shape[0]
                for i in tqdm(range(n_channel), desc=name, leave=False):
                    paramGradSub = paramGrad[i] if paramGrad is not None else None
                    paramHessSub = paramHess[i] if paramHess is not None else None
                    param.data[i] = quantLayer_with_MSE(device, bit, args, name, param[i], power=power, grad=paramGradSub, hess=paramHessSub, channel_wise=args.run_channelWise)
            else:
                param.data = quantLayer_with_MSE(device, bit, args, name, param, power=power, grad=paramGrad, hess=paramHess, channel_wise=args.run_channelWise)
                
    evalLog = dict()
    eval(quant_net, test_loader, device, train_args.criterion, 0, log = evalLog)
    return evalLog['acc']

def quantLayer_with_KLDiv(device, bit, args, name, param, power=1.0, grad=None, hess=None):
    paramBefore = param.clone().detach()
    paramAfter  = param.clone().detach()

    #Quantize resolution
    clipStart = args.quant_clip_start
    clipEnd   = args.quant_clip_end
    quant_resolution = args.quant_resolution
    clipSize  = clipEnd - clipStart
    clipStep  = clipSize / quant_resolution
    
    #Find Minimum point
    minClipping = 1.0
    minKlDiv = 1e+100
    for clp in range(1, quant_resolution+1):
        clipping = clipStart + clipStep*clp
        quantDict = dict()
        paramAfter.data = weightQuantize(name, param, bit, clipping, quantDict=quantDict)
        curKlDiv = klDivParam(paramBefore, paramAfter, bit, quantDict['max'])
        if curKlDiv < minKlDiv: #update minimum point
            minClipping = clipping
            minKlDiv = curKlDiv
    return weightQuantize(name, param, bit, minClipping)

def quantModel_with_KLDiv(device, bit, args, test_loader, train_args):
    quant_net = getModel(args)
    quant_net.load_state_dict(torch.load(args.load_param_path, map_location=device))
    quant_net.to(torch.device(device))
    for name, param in quant_net.named_parameters():
        if re.search(r'conv.\.weight', name):
            if args.run_channelWise:
                n_channel = param.data.shape[0]
                for i in tqdm(range(n_channel), desc=name, leave=False):
                    param.data[i] = quantLayer_with_KLDiv(device, bit, args, name, param[i])
            else:
                param.data = quantLayer_with_KLDiv(device, bit, args, name, param)
    evalLog = dict()
    eval(quant_net, test_loader, device, train_args.criterion, 0, log = evalLog)
    return evalLog['acc']
    
def quantModel_with_minMax(device, bit, args, test_loader, train_args):
    quant_net = getModel(args)
    quant_net.load_state_dict(torch.load(args.load_param_path, map_location=device))
    quant_net.to(torch.device(device))
    for name, param in quant_net.named_parameters():
        if re.search(r'conv.\.weight', name):
            param.data = weightQuantize(name, param, bit, 1.0)
    evalLog = dict()
    eval(quant_net, test_loader, device, train_args.criterion, 0, log = evalLog)
    return evalLog['acc']
            
def quantChannel(device, bit, test_loader, train_args, args):
    log = dict()
    log['mseSumP1.0'] = quantModel_with_MSE(device, bit, args, test_loader, train_args, power=1.0, useGrad=False, useHess=False)
    log['mseSumP2.0'] = quantModel_with_MSE(device, bit, args, test_loader, train_args, power=2.0, useGrad=False, useHess=False)
    for i in tqdm(range(12), desc='grad level'):
        grad = 1.0 + (i * 0.5)
        log[f'gradP{grad:2.1f}']   = quantModel_with_MSE(device, bit, args, test_loader, train_args, power=grad, useGrad=True , useHess=False)
    log['klDiv']  = quantModel_with_KLDiv (device, bit, args, test_loader, train_args)
    log['minMax'] = quantModel_with_minMax(device, bit, args, test_loader, train_args)
    # for i in tqdm(range(5)):
    #     grad = 1.0 + (i * 0.2)
    #     log[f'hessP{grad:2.1f}']   = quantModel_with_MSE(device, bit, args, test_loader, train_args, power=grad, useGrad=True , useHess=True)
    return log
    