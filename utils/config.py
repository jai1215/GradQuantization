PATH = './cifar_net.pth'
# LOADPATH = 'cifar_resnet18.pth'
# LOADPATH = 'cifar_net.pth'
LOADPATH = f"./checkPoints/resnet18_weight_test.pth"
GRADPATH = f"./checkPoints/resnet18_grad.pth"
LOAD = True

JSONPATH = f'./jsons/'

maxEpoch = 1

batch_size = 1
batch_size_test = 100
num_worker = 4
mnist=False
fuseModel=False
run_traing=False
run_eval=False
run_grad=False
run_hessian=False
save_training=False