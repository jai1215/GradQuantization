import os
import torch
import argparse

from dataSet import getDataLoader
from mainFunctions import *

#my code
from gradQuantize import generateGrad
from gradQuantize import quantEval
from gradQuantize import quantChannel

import pickle
from tqdm import tqdm

from utils.utilFuncs import *
from train import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='gradQuant Parameters')
    # Project Setting
    parser.add_argument('--TEST', default='resnet18', type=str, help='project name')
    
    parser.add_argument('--seed', default=2022, type=int, help='random seed for result reproducing')
    parser.add_argument('--arch', default='resnet18', type=str, help='dataset name',
                    choices=['resnet18', 'resnet50'])
    parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size for data loader')
    parser.add_argument('--workers', default=4, type=int, help='number of workers for data loader')
    parser.add_argument('--data_path', default='../data', type=str, help='path to CIFAR10Net data', required=False)
    
    # Training Parameters
    parser.add_argument('--max_epoch', default=1, type=int, help='trainning epoch')
    
    # Data Loading
    parser.add_argument('--load_param'      , default=True, type=bool, help='Load pre trained parameter')
    parser.add_argument('--load_param_path' , default='./data/model/resnet18_weight_test.pth', type=str, help='pretrained weight')
    
    # Make Grad Data
    parser.add_argument('--grad_epoch'      , default=16, type=int, help='run traing to make weights')
    parser.add_argument('--dump_path'       , default='./data/resnet18_cifar/weights', type=str, help='weight adn gradient dump path')
    
    # Quant Options
    parser.add_argument('--qaunt_symmetric' , default=False, type=bool, help='weight symmetric quantization')
    
    # Grad Quantization
    parser.add_argument('--quant_resolution'  , default=100, type=int,     help='quantization resolution')
    parser.add_argument('--quant_clip_start'  , default=0.0, type=float,   help='quantization clipping start')
    parser.add_argument('--quant_clip_end'    , default=1.0, type=float,   help='quantization clipping end')
    parser.add_argument('--quant_bit_start'   , default=1,   type=int,     help='quantization bit start')
    parser.add_argument('--quant_bit_end'     , default=8,   type=int,     help='quantization bit end')
    parser.add_argument('--quant_run_eval'    , default=False, type=bool,  help='quantization run eval data')
    parser.add_argument('--quant_use_hess'    , default=False, type=bool,  help='quantization use hessian')
    parser.add_argument('--quant_base_data'   , default=False, type=bool,  help='quantization save result as base data')
    parser.add_argument('--quant_result_path' , default='./data/resnet18_cifar/result/', type=str,  help='quantization save path')
    
    # Quantization Flow Control
    parser.add_argument('--run_train'      , default=False, type=bool, help='run traing to make weights')
    parser.add_argument('--run_grad'       , default=False, type=bool, help='run gradient dump')
    parser.add_argument('--run_hess'       , default=False, type=bool, help='run hessian  dump')
    parser.add_argument('--run_32fp_eval'  , default=False, type=bool, help='run 32bit fp evaluation')
    parser.add_argument('--run_layerWise'  , default=True , type=bool, help='run grad Matric')
    parser.add_argument('--run_channelWise', default=True , type=bool, help='run channel wise grad Matric')
    
    # Server Configuration
    parser.add_argument('--gpu', default='cuda:0', type=str, help='select gpu')
    
    args = parser.parse_args()
    seed_all(args.seed)
    
    ## -- Select Device / Data / Model
    device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu') if args.gpu[:4] == 'cuda' else 'cpu'
    train_loader, test_loader = getDataLoader(args)
    model = getModel(args)

    ## -- Loading Model Parameter
    if args.load_param:
        print("Loading model from: ", args.load_param_path)
        if device == 'cpu':
            model.load_state_dict(torch.load(args.load_param_path, map_location=device))
        else:
            model.load_state_dict(torch.load(args.load_param_path, map_location=device))
    model.to(device)
    
    train_args = getTrainArgs(args, model)

    ## -- Training
    if args.run_train:
        print("Main : Running Training")
        bestAcc = 0.0
        for epoch in range(args.max_epoch):   ## Epoch
            train_loss = train(model, train_loader, device, train_args.optimizer, train_args.criterion)
            bestAcc    = eval (model,  test_loader, device, train_args.criterion, epoch, bestAcc)
        if save_training:
            os.replace(args.load_param_path, args.load_param_path+getToday()) #move old parameter
            torch.save(model.state_dict(), args.load_param_path)
        exit(0)
            
    ## -- get Gradient
    if args.run_grad or args.run_hess:
        print("Main : Generating Gradient")
        generateGrad(model, train_loader, device, train_args.criterion, args, train_args.optimizer)
        exit(0)

    ## -- 32b float Evaluation
    if run_eval:
        print("Main : Running Evaluation")
        eval_net = getModel(args)
        eval_net.load_state_dict(torch.load(args.load_param_path))
        eval(eval_net, test_loader, device, train_args.criterion, bestAcc)
        exit(0)
        
    ### --- Grad Quantization Using Matrics
    if args.run_layerWise:
        print("Starting Layer-wise quantization")
        dumpID = getToday()
        Logs = {'config': {}, 'data':[]}
        Logs['config']['args'] = args
        for bit in range(args.quant_bit_start, args.quant_bit_end):
            print("Quantizing in bit %d" % bit)
            with torch.no_grad():
                Logs[f'bit_{bit}'] = quantChannel(device, bit, test_loader, train_args, args)
        with open('run_channel.pkl', 'wb') as f:
            print(f"Dump : result is dumping in run_channel.pkl")
            pickle.dump(Logs ,f)
        exit(0)
        
    ### --- Grad Quantization Scan All Step Code
    print("Starting Grad Quantization")
    dumpID = getToday()
    quant_resolution = args.quant_resolution
    for bits in range(args.quant_bit_start, args.quant_bit_end):
        print("Quantizing in bits %d" % bits)
        Logs = {'config': {}, 'data':[]}
        Logs['config']['bits'] = bits
        Logs['config']['resolution'] = quant_resolution
        Logs['config']['args'] = args
        with torch.no_grad():
            clipStart = args.quant_clip_start
            clipEnd   = args.quant_clip_end
            clipSize  = clipEnd - clipStart
            clipStep  = clipSize / quant_resolution
            for clp in tqdm(range(1, quant_resolution+1)):
                log = {}
                log['clipping'] = clipStart + clipStep*clp
                
                quant_net = getModel(args)
                quant_net.load_state_dict(torch.load(args.load_param_path, map_location=device))
                quant_net.to(torch.device(device))
                quant_net = quantEval(device, quant_net, bits, log['clipping'], log, args, symmetric=True, useHess=args.quant_use_hess)
                if args.quant_run_eval:
                    eval(quant_net, test_loader, device, train_args.criterion, 0, log=log)
                Logs['data'].append(log)

        dumpPath = args.quant_result_path
        dumpPath += ('./baseData_' if args.quant_base_data else '/test_')
        dumpPath += f'{bits}_{dumpID}.pkl'
        with open(dumpPath, 'wb') as f:
            print(f"Dump : result is dumping in {dumpPath}")
            pickle.dump(Logs ,f)
