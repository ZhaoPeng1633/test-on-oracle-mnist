import argparse
import torch, os, gzip
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import mnist_reader

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=300, metavar='N',
                    help='input batch size for testing (default: 300)')
parser.add_argument('--epochs', type=int, default=15, metavar='N', help='number of epochs to train (default: 15)')
parser.add_argument('--net', type=str, default='Net3', choices=["Net1", "Net2", "Net3"], help='type of network')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
parser.add_argument('--data-dir', type=str, default='../data/oracle/', help='data path')
parser.add_argument('--use-cuda', action='store_true', default=False, help='CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-model', action='store_true', default=True, help='For Saving the current Model')

args = parser.parse_args()
train_data = ImageList(path=args.data_dir, kind='train',
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,))
                           ]))
train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)

test_data = ImageList(path=args.data_dir, kind='t10k',
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,))
                          ]))
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, **kwargs)




for i in range(20, 50):
    image_out = test_image[i]
    image_out = torch.unsqueeze(image_out, dim=0)
    net.load_state_dict(torch.load('minst_cnn.pt'))
    with torch.no_grad():
        out = net(image_out)
        predict_name = torch.max(out, dim=1)[1].data.numpy()
        predict = torch.softmax(out, dim=1)
        print('图片序号 %d %s 预测概率为' % (i, train_data.classes[int(predict_name)]))
        for j in range(10):
            print('%s:%.5f' %(train_data.classes[j], predict[0][j]), end=" ")
        print("\n")