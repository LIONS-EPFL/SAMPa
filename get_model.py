from model.resnet import resnet56
from model.wide_res_net import WideResNet
from model.densenet import DenseNet121
from model.vgg import vgg19_bn

def get_model(args):
    if args.dataset == 'cifar10':
        num_classes = 10  
    elif args.dataset == 'cifar100':
        num_classes = 100
        
    if args.model == 'wrn2810':
        model = WideResNet(28, 10, args.dropout, in_channels=3, labels=num_classes)
    elif args.model == 'wrn282':
        model = WideResNet(28, 2, args.dropout, in_channels=3, labels=num_classes)
    elif args.model == 'resnet56':
        model = resnet56(num_classes=num_classes)
    elif args.model == 'densenet121':
        model = DenseNet121(num_classes=num_classes) 
    elif args.model == 'vgg19bn':
        model = vgg19_bn(num_classes=num_classes)  

    return model