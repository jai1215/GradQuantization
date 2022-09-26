import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os, re, argparse, pickle

from drawLabels import *

def latestDumps(base, path):
    files = os.listdir(path)
    p = r'%s_(\d)_(.+)\.pkl' % base
    
    ret = dict()
    
    for f in files:
        m = re.match(p, f)
        key = int(m.group(1))
        value = m.group()
        #update ret
        if key in ret:
            if value > ret[key]:
                ret[key] = value
        else:
            ret[key] = value
    return ret
    
def loadPickle(f):
    ret = None
    with open(f, 'rb') as fin:
        ret = pd.DataFrame(pickle.load(fin)['data'])
    return ret

if __name__ == '__main__':
    # Temporal Code - begin
    TEST = True
    if TEST:
        ret = None
        with open('run_channel.pkl', 'rb') as fin:
            ret = pickle.load(fin)
            
        lines = []
        for i in range(1, 8):
            df_sub = pd.DataFrame([ret[f'bit_{i}']])
            df_sub['bit'] = f'bit_{i}'
            lines.append(df_sub)
        df = pd.concat(lines)
        df.set_index('bit', inplace=True)
        
        print(df)
        with open(f'result.html', 'wt') as fout:
            fout.write(df.to_html())
            
        exit(1)
    #Temporal Code - end
    
    
    
    parser = argparse.ArgumentParser(description='dara Plot Parameters')
    parser.add_argument('--TEST'              , default='resnet18', type=str, help='project name')
    parser.add_argument('--quant_result_path' , default='./data/resnet18_cifar/result/' , type=str,  help='quantization save path')
    parser.add_argument('--dump_html_path'    , default='./data/resnet18_cifar/htmls/'  , type=str,  help='html dump path')
    parser.add_argument('--dump_png_path'     , default='./data/resnet18_cifar/png/'    , type=str,  help='png dump path')
    parser.add_argument('--dump_html'         , default=True  , type=bool,  help='html dump enable')
    parser.add_argument('--dump_png'          , default=True   , type=bool,  help='png dump enable')
    parser.add_argument('--dump_bits'         , default='7', type=str,  help='dumpping bits')
    parser.add_argument('--dump_acc_html'     , default=False, type=bool,  help='dump acc html')
    parser.add_argument('--minLossArea'       , default=True , type=bool,  help='only print minmize loss area +-0.15')
    
    args = parser.parse_args()
    
    ## -- Select Bit
    bit = int(args.dump_bits)
    
    ## -- Data Loading
    basePaths = latestDumps('baseData', args.quant_result_path)
    df = loadPickle(args.quant_result_path + basePaths[bit])
    df.set_index('clipping', inplace=True)
    
    # 충돌범위
    # conflictSet = set(mse_df.columns) & set(loss_df.columns) 
    # loss_df = loss_df.drop(conflictSet, axis=1)
    # df = loss_df.join(mse_df)
    # df = df.interpolate(method="linear")
    
    ## -- Print as HTML
    if args.dump_html:
        with open(f'{args.dump_html_path}/rawData_{bit}.html', 'wt') as fout:
            fout.write(df.to_html())
            
    ## -- Finding Maxmimum Accuracy
    min_idx = df.idxmin()
    acc_df = df['acc'][min_idx]
    acc_df.index  = min_idx.index.tolist()
    acc_df = acc_df.sort_values(ascending=False)
    acc_df = pd.DataFrame(acc_df).join(pd.DataFrame(min_idx))
    
    print("Top Accuracy")
    print(acc_df[:10])
    
    if args.dump_acc_html:
        dump_path = f'{args.dump_html_path}/acc_{bit}.html'
        with open(dump_path, 'wt') as fout:
            print(f"Dump : accuracy Data in {dump_path}")
            fout.write(acc_df.to_html())
    
    ## -- minimize Loss
    if args.minLossArea:
        minimizeLoss = df['loss'].idxmin()
        df = df[df.index >= minimizeLoss - 0.15]
        df = df[df.index <= minimizeLoss + 0.15]
    
    ## -- Normalize df
    df = (df - df.min()) / (df.max() - df.min())
    
    ## -- Load Print Labels
    labels = ['loss']
    labels += drawLabels
    
    # plt.subplot(2, 2, plotIdx+1)
    # sns.lineplot(x='clipping', y='acc', data=df, lw=4, legend='brief', label='acc')
    sns.lineplot(x='clipping', y='loss', data=df, lw=3, legend='brief', label='loss', linestyle='--', color='black')
    for l in labels[1:]:
        sns.lineplot(x='clipping', y=l, data=df, legend='brief', label=l)
    plt.xlabel("Clipping Range", fontsize = 10)
    plt.ylabel("Loss(Normalized)", fontsize = 10)
    plt.title(f"{bit} bits quantization loss")
    
    if args.dump_png:
        dump_path = f'{args.dump_png_path}/dumpPLT_{bit}.png'
        print(f"Dump : plot is stored in {dump_path}")
        plt.savefig(dump_path)
    
    plt.show()
    
# comp_df = pd.DataFrame(0, index=drawBits, columns=labels)

# plt.figure(figsize=(16,6))
# for plotIdx, bits in enumerate(drawBits):
#     ##-- Load dataframe    
#     loss_df    = loadJson(JSONPATH+f'loss_{bits}.json')
#     mse_df     = loadJson(JSONPATH+f'result_bit_{bits}.json')

#     ##-- Set Index
#     loss_df = loss_df.set_index('clipping')
#     mse_df = mse_df.set_index('clipping')

#     conflictSet = set(mse_df.columns) & set(loss_df.columns) 
#     loss_df = loss_df.drop(conflictSet, axis=1)
#     df = loss_df.join(mse_df)

#     # with open('mse.html', 'wt') as fout:
#     #     fout.write(mse_df.to_html())

#     # df['loss'] = np.log(df['loss']) #Log Scale

#     ##-- Interpolate 결손값
#     df = df.interpolate(method="linear")

#     with open(f'./htmls/mse_{bits}.html', 'wt') as fout:
#         fout.write(df.to_html())

#     ## -- Finding Maxmimum Accuracy
#     idx = df.idxmin()
#     acc = df['acc'][idx]
#     acc.index  = idx.index.tolist()
#     acc = acc.sort_values(ascending=False)
#     acc = pd.DataFrame(acc).join(pd.DataFrame(df.idxmin()))
#     for i in range(1, 16):
#         acc.drop(index = f'gradP1.0_{i}', inplace=True)
#         acc.drop(index = f'gradP1.5_{i}', inplace=True)
#         acc.drop(index = f'gradP2.0_{i}', inplace=True)
#     print(acc[:20])
#     with open(f'./htmls/acc_{bits}.html', 'wt') as fout:
#         fout.write(acc.to_html())
    
#     comp_label = list(labels) + ['gradP1.2','gradP1.4','gradP1.7','gradP2.0','gradP2.5','gradP3.0']
#     for l in comp_label:
#         if l in acc.index:
#             comp_df.loc[bits, l] = acc.loc[l, 'acc']
#         else:
#             comp_df.loc[bits, l] = comp_df.loc[bits-1, l]
#     dat = df['acc'][1.0]
#     comp_df.loc[bits, 'min-max'] = dat
#     minimizeLoss = df['loss'].idxmin()
#     #Normalize

#     df = df[df.index >= minimizeLoss - 0.15]
#     df = df[df.index <= minimizeLoss + 0.15]
#     # df = df[df.index >= 0.5]
#     # df = df[df.index <= 0.8]
    
#     df = (df - df.min()) / (df.max() - df.min())
#     #
#     #loss delta
#     # plt.subplot(1, 2, 2)
#     # df['clipError'] = df['mseGrad20'] - df['roundGrad20']
#     # sns.lineplot(x='clipping', y='roundGrad20', data=df, legend='brief', label='roundGrad20')
#     # sns.lineplot(x='clipping', y='mseGrad20', data=df, legend='brief', label='mseGrad20')
#     # sns.lineplot(x='clipping', y='clipError', data=df, legend='brief', label='clipError')

#     # for i in range(1, 16):
#     #     labels.append('gradP1.5_%d' % i)

#     plt.subplot(2, 2, plotIdx+1)
#     # sns.lineplot(x='clipping', y='acc', data=df, lw=4, legend='brief', label='acc')
#     sns.lineplot(x='clipping', y='loss', data=df, lw=3, legend='brief', label='loss', linestyle='--', color='black')
#     for l in labels[1:]:
#         sns.lineplot(x='clipping', y=l, data=df, legend='brief', label=l)
#     plt.xlabel("Clipping Range", fontsize = 10)
#     plt.ylabel("Loss(Normalized)", fontsize = 10)
#     plt.title(f"{bits} bits quantization loss")
    

# comp_df = comp_df.reindex(sorted(comp_df.columns), axis=1)
# print(comp_df)

# #-- Plot Accuracy
# plt.figure(figsize=(16,6))
# comp_df = np.log10(comp_df) #Log Scale

# # plt.yscale('log')
# for l in comp_label:
#     sns.lineplot(x=comp_df.index[3:], y=l, data=comp_df.iloc[3:], legend='brief', label=l, markers=True)
# # plt.ylim([85,94])
# # plt.xlim([3, 8])

# plt.show()
