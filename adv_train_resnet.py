from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from wideresnet import *
from resnet import *
from mart import mart_loss, modified_mart_loss, normalize
import numpy as np
import time

from models.resnet import *




# os.environ["CUDA_VISIBLE_DEVICES"]="0"

parser = argparse.ArgumentParser(description='PyTorch CIFAR MART Defense')
parser.add_argument('--attack', default='pgd')
parser.add_argument('--oversample', action='store_true', default=False,
                    help='oversample training data')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=120, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=3.5e-3,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.031,
                    help='perturbation')
parser.add_argument('--num-steps', default=10,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.007,
                    help='perturb step size')
parser.add_argument('--beta', default=5.0,
                    help='weight before kl (misclassified examples)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model', default='resnet',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')

args = parser.parse_args()


model_dir = "results/" + args.attack + "/"

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target]) 

class Batches():
    def __init__(self, dataset, batch_size, shuffle=False, set_random_choices=False, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.set_random_choices = set_random_choices
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, **kwargs
        )

    def __iter__(self):
        if self.set_random_choices:
            self.dataset.set_random_choices()
        return ({'input': x.to(device).float(), 'target': y.to(device).long()} for (x,y) in self.dataloader)

    def __len__(self):
        return len(self.dataloader)


# setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
train_set = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
test_set = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)

if args.attack == "all" :
    train_data = np.array(train_set.data) / 255.
    train_data = transpose(train_data).astype(np.float32)

    train_labels = np.array(train_set.targets)

    oversampled_train_data = train_data.copy()
    oversampled_train_labels = train_labels.copy()

    for i in range(10) :
        oversampled_train_data = np.concatenate((oversampled_train_data, train_data))
        oversampled_train_labels = np.concatenate((oversampled_train_labels, train_labels))

    # print("Shape")
    # print(oversampled_train_data.shape)
    # print(oversampled_train_labels.shape)
    train_set = list(zip(torch.from_numpy(oversampled_train_data), torch.from_numpy(oversampled_train_labels)))


train_batches = Batches(train_set, args.batch_size, shuffle=True)
test_batches = Batches(test_set, args.batch_size, shuffle=False)


# load adversarial examples

train_adv_images = None
train_adv_labels = None
test_adv_images = None
test_adv_labels = None


if args.attack != "all" :
    adv_dir = "adv_examples/{}/".format(args.attack)
    train_path = adv_dir + "train.pth" 
    test_path = adv_dir + "test.pth"

    adv_train_data = torch.load(train_path)
    train_adv_images = adv_train_data["adv"]
    train_adv_labels = adv_train_data["label"]

    adv_test_data = torch.load(test_path)
    test_adv_images = adv_test_data["adv"]
    test_adv_labels = adv_test_data["label"]
else :
    ATTACK_LIST = ["autoattack", "autopgd", "bim", "cw", "deepfool", "fgsm", "newtonfool", "pgd", "pixelattack", "spatialtransformation", "squareattack"]
    for i in range(len(ATTACK_LIST)):
        _adv_dir = "adv_examples/{}/".format(ATTACK_LIST[i])
        train_path = _adv_dir + "train.pth" 
        test_path = _adv_dir + "test.pth"

        adv_train_data = torch.load(train_path)
        adv_test_data = torch.load(test_path)

        if i == 0 :
            train_adv_images = adv_train_data["adv"]
            train_adv_labels = adv_train_data["label"]
            test_adv_images = adv_test_data["adv"]
            test_adv_labels = adv_test_data["label"]   
        else :
            train_adv_images = np.concatenate((train_adv_images, adv_train_data["adv"]))
            train_adv_labels = np.concatenate((train_adv_labels, adv_train_data["label"]))
            test_adv_images = np.concatenate((test_adv_images, adv_test_data["adv"]))
            test_adv_labels = np.concatenate((test_adv_labels, adv_test_data["label"]))

print("Shape")
print(train_adv_images.shape)
print(test_adv_images.shape)

train_adv_set = list(zip(train_adv_images,
    train_adv_labels))

train_adv_batches = Batches(train_adv_set, args.batch_size, shuffle=False)

test_adv_set = list(zip(test_adv_images,
    test_adv_labels))

test_adv_batches = Batches(test_adv_set, args.batch_size, shuffle=False)


def train(args, model, device, train_batches, train_adv_batches, optimizer, epoch):
    model.train()
    for batch_idx, (batch, adv_batch) in enumerate(zip(train_batches, train_adv_batches)):
        data = batch["input"]
        target = batch["target"]
        
        x_adv = adv_batch["input"]
        y_adv = adv_batch["target"]
   
        optimizer.zero_grad()

        # calculate robust loss
        loss = modified_mart_loss(model=model,
                           x_natural=data,
                           x_adv=x_adv,
                           y=target,
                           y_adv = y_adv,
                           optimizer=optimizer,
                           step_size=args.step_size,
                           epsilon=args.epsilon,
                           perturb_steps=args.num_steps,
                           beta=args.beta)
        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_batches.dataset),
                       100. * batch_idx / len(train_batches), loss.item()))

def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 100:
        lr = args.lr * 0.001
    elif epoch >= 90:
        lr = args.lr * 0.01
    elif epoch >= 75:
        lr = args.lr * 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
        
def evaluate(model, device, data_loader):
    model.eval()
    eval_loss = 0
    correct = 0
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            data, target = batch["input"], batch["target"]
            output = model(normalize(data))
            eval_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    eval_loss /= len(data_loader.dataset)
    accuracy = correct / len(data_loader.dataset)
    return eval_loss, accuracy


def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=20,
                  step_size=0.003):
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)

    random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
    X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
    return err, err_pgd

def eval_adv_test_whitebox(model, device, test_loader):

    model.eval()
    robust_err_total = 0
    natural_err_total = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust = _pgd_whitebox(model, X, y)
        robust_err_total += err_robust
        natural_err_total += err_natural
    print('natural_acc: ', 1 - natural_err_total / len(test_loader.dataset))
    print('robust_acc: ', 1- robust_err_total / len(test_loader.dataset))
    return 1 - natural_err_total / len(test_loader.dataset), 1- robust_err_total / len(test_loader.dataset)


def main():

#     model = ResNet18().to(device)

    model = resnet18(pretrained=True)
    model.cuda()
    

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)   
    
    natural_acc = []
    robust_acc = []
    
    best_test_robust_acc = 0
    
    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)
        
        start_time = time.time()

       # adversarial training
        train(args, model, device, train_batches, train_adv_batches, optimizer, epoch)

        # evaluation on natural examples
        print('==============')
        _, train_accuracy = evaluate(model, device, train_batches)
        _, test_accuracy = evaluate(model, device, test_batches)
        _, train_robust_accuracy = evaluate(model, device, train_adv_batches)
        _, test_robust_accuracy = evaluate(model, device, test_adv_batches)

        print("Train Accuracy: ", train_accuracy)
        print("Test Accuracy: ", test_accuracy)
        print("Train Robust Accuracy: ", train_robust_accuracy)
        print("Test Robust Accuracy: ", test_robust_accuracy)
        print('==============')

        
        # save best
        if test_robust_accuracy > best_test_robust_acc:
            torch.save({
                    'state_dict':model.state_dict(),
                    'test_robust_acc':test_robust_accuracy,
                    'test_accuracy':test_accuracy,
                }, os.path.join(model_dir, f'model_best.pth'))
            best_test_robust_acc = test_robust_accuracy
        
        # save checkpoint
#         if epoch % args.save_freq == 0:
#                 torch.save(model.state_dict(), os.path.join(model_dir, 'res18-epoch{}.pt'.format(epoch)))


if __name__ == '__main__':
    main()
