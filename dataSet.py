import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

def getDataLoader(args):
	T_mean = (0.4914, 0.4822, 0.4465)
	T_std = (0.2471, 0.2435, 0.2616)
	transforms_train = transforms.Compose([
		transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		# transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		transforms.Normalize(T_mean, T_std),
	])
	# transforms_test = transforms.Compose([
	# 	transforms.ToTensor(),
	# 	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	# ])
	transforms_test = transforms.Compose(
	[
		transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize(T_mean, T_std),
	]
	)

	dataset_train = CIFAR10(root=args.data_path, train=True, 
							download=args.data_download, transform=transforms_train)
	dataset_test = CIFAR10(root=args.data_path, train=False, 
						download=args.data_download, transform=transforms_test)

	train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,  num_workers=args.workers)
	test_loader = DataLoader(dataset_test, batch_size=args.batch_size, 	 shuffle=False, num_workers=args.workers)
	return train_loader, test_loader

    # if args.mnist:
    # 	dataset_train = DataLoader(torchvision.datasets.MNIST('../data', train=True, download=True,
	# 													transform=transforms.Compose([transforms.ToTensor(),
	# 																		transforms.Normalize((0.1307,), (0.3081,))
	# 													])),
	# 				batch_size=64, shuffle=True, num_workers=1, pin_memory=True)

	# 	dataset_test = DataLoader(torchvision.datasets.MNIST('../data', train=False, 
	# 														transform=transforms.Compose([transforms.ToTensor(),
	# 																			transforms.Normalize((0.1307,), (0.3081,))
	# 														])),
	# 				batch_size=64, shuffle=True, num_workers=1, pin_memory=True)
