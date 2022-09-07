from collections import namedtuple
import torch
import torch.nn as nn

QTensor = namedtuple('QTensor', ['tensor', 'scale', 'zero_point'])

def calcScaleZeroPoint(min_val, max_val, num_bits=8):
    # Calc Scale and zero point of next 
    qmin = 0.
    qmax = 2.**num_bits - 1.
    
    scale = (max_val - min_val) / (qmax - qmin)
    initial_zero_point = qmin - min_val / scale
    
    zero_point = 0
    if initial_zero_point < qmin:
        zero_point = qmin
    elif initial_zero_point > qmax:
        zero_point = qmax
    else:
        zero_point = initial_zero_point

    zero_point = int(zero_point)

    return scale, zero_point

def calcScaleZeroPointSym(min_val, max_val,num_bits=8):
    # Calc Scale 
    max_val = max(abs(min_val), abs(max_val))
    qmin = 0.
    qmax = 2.**(num_bits-1) - 1.

    scale = max_val / qmax

    return scale, 0

def quantize_tensor(x, num_bits=8, min_val=None, max_val=None):
    if not min_val and not max_val: 
        min_val, max_val = x.min(), x.max()

    qmin = 0.
    qmax = 2.**num_bits - 1.

    scale, zero_point = calcScaleZeroPoint(min_val, max_val, num_bits)
    q_x = zero_point + x / scale
    q_x.clamp_(qmin, qmax).round_()
    q_x = q_x.round().byte()
    
    return QTensor(tensor=q_x, scale=scale, zero_point=zero_point)

def dequantize_tensor(q_x):
    return q_x.scale * (q_x.tensor.float() - q_x.zero_point)

def quantize_tensor_sym(x, min_val=None, max_val=None, num_bits=8):
    #retain only input channel
    if (min_val is None) or (max_val is None):
        max_val, _ = torch.max(x)
        min_val, _ = torch.min(x)
        
    max_val = max(abs(min_val), abs(max_val))
    qmax = 2.**(num_bits-1) - 1.
    scale = max_val / qmax
    q_x = x/scale
    q_x.clamp_(-qmax, qmax).round_()
    q_x = q_x.round()
    
    #temp code
    x = q_x * scale
    return x

def minmax_with_channel(x, remain_channel=-1):
    if remain_channel == -1:
        return torch.min(x), torch.max(x)
    
    max_val = x
    min_val = x
    dim = max_val.dim()
    for d in range(dim-1, -1, -1):
        if d == remain_channel:
            continue
        max_val, _ = torch.max(max_val, dim=d)
        min_val, _ = torch.min(min_val, dim=d)
    return min_val, max_val
        
def quantize_weight_sym(x, num_bits=8):
    if x is None:
        return None
    min_val, max_val = minmax_with_channel(x, remain_channel=0)
    max_val = torch.max(torch.abs(min_val), torch.abs(max_val))
    qmax = 2.**(num_bits-1) - 1.
    scale = max_val / qmax
    #Transpose for division and re-transpose
    xDim = x.dim()
    x = x.transpose(0, xDim-1)
    q_x = torch.div(x, scale)
    q_x = q_x.transpose(0, xDim-1)

    #Clamps and rounds
    q_x.clamp_(-qmax, qmax).round_()
    q_x = q_x.round()

    #Transpose for multiplication and re-transpose(temp code)
    q_x = q_x.transpose(0, xDim-1)
    x = q_x * scale
    x = x.transpose(0, xDim-1)
    return x

def dequantize_tensor_sym(q_x):
    return q_x.scale * (q_x.tensor.float())

def updateRangeSub(x, key, actRange):
    min_val, max_val = minmax_with_channel(x, remain_channel=0)
    max_val = torch.mean(max_val)
    min_val = torch.mean(min_val)
    
    if key not in actRange:
        actRange[key] = {'max': max_val, 'min': min_val, 'total':1}
    else:
        actRange[key]['max'] += max_val
        actRange[key]['min'] += min_val
        actRange[key]['total'] += 1
            
def qConv2d(conv, x, num_bits=8):
    W = conv.weight.data
    B = None if (conv.bias is None) else conv.bias.data
    conv.weight.data = quantize_weight_sym(W, num_bits=num_bits)
    if B is not None:
        conv.bias.data = quantize_weight_sym(B, num_bits=num_bits)
    #Run Foward
    return conv(x)

def qAct(x, actRange, key, sym=False, num_bits=8):
    return quantize_tensor_sym(x, actRange[key]['min_val'], actRange[key]['max_val'], num_bits=num_bits)

def qLinear(conv, x, num_bits=8):
    W = conv.weight.data
    B = None if (conv.bias is None) else conv.bias.data
    conv.weight.data = quantize_weight_sym(W, num_bits=num_bits)
    if B is not None:
        conv.bias.data = quantize_weight_sym(B, num_bits=num_bits)
    #Run Foward
    return conv(x)