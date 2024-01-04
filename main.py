import argparse
import os
import random
import shutil
import time
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from utils import get_train_dataset, get_default_val_loader, get_default_train_sampler_loader, accuracy
from model import get_ConTrans_func_by_name


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

data_TRAINSET_SIZE = 733        


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='ConTrans', help='name of model')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=1, type=int, metavar='N',      
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,             
                    metavar='N')
parser.add_argument('--val-batch-size', default=16, type=int, metavar='V',     
                    help='validation batch size')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0, type=float,         
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

best_acc1 = 0
torch.backends.cudnn.benchmark = True


def main():
    args = parser.parse_args()
    ngpus_per_node = torch.cuda.device_count()
    main_worker(ngpus_per_node, args)

def main_worker(ngpus_per_node, args):
    global best_acc1

    ConTrans_build_func = get_ConTrans_func_by_name(args.arch)
    model = ConTrans_build_func(deploy=False)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr,  betas=(0.9,0.999), eps=1e-8, weight_decay=args.weight_decay)

    cudnn.benchmark = True

    train_sampler, train_loader = get_default_train_sampler_loader(args)
    val_loader = get_default_val_loader(args)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    time_open = time.time()
    global_acc = 0.
    global_acc_train = 0.
    global_loss_train = 10.0
    for epoch in range(args.start_epoch, args.epochs+1):        
        train(train_loader, model, criterion, optimizer, epoch, args)
        
        best_acc = validate(epoch, val_loader, model, criterion, args)
        if global_acc < best_acc:
            print('==> Saving model.......')
            global_acc = best_acc
            torch.save(model.state_dict(), '   ')     #  model directory
        
    time_end = time.time() - time_open
    print('Total time:', time_end)


training_loss, valid_loss = [], []
Train_acc_list, valid_acc_list = [],[]

def train(train_loader, model, criterion, optimizer, epoch, args):

    start = time.perf_counter()
    model.train()

    train_loss = 0.
    acc = 0.
    total = 0.
    train_acc1 = 0.


    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.to(device)
        target = target.to(device)

        # output
        output = model(images)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, pred = output.max(1)
        total += target.size(0)
        acc += pred.eq(target).sum().item()
        train_acc = 100 * acc / total

        if i % args.print_freq == 0:
            print('[Train Epoch: {} [{}/{} ({:.0f}%)] ], Loss1: {:.6f}, Loss2: {:.6f}, Train Acc: {:.5f}%, Correct {} / Total {}'.format(
                    epoch,
                    i * args.batch_size, len(train_loader.dataset),
                    100. * i / len(train_loader),
                    loss.item(),
                    train_loss / (i + 1),
                    train_acc,
                    acc, total))

        end = time.perf_counter()

    training_loss.append(loss.item())
    Train_acc_list.append(train_acc)


def validate(epoch, val_loader, model, criterion, args):

    model.eval()

    val_loss = 0
    total_num = 0
    acc = 0.

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images, target = images.to(device), target.to(device)

            # output
            output = model(images)
            loss = criterion(output, target)

            val_loss += loss
            _, pred = output.max(1)
            acc += pred.eq(target).sum().item()
            total_num += target.size(0)
            val_loss /= len(val_loader.dataset)
            test_acc = acc / total_num

            batch_time.update(time.time() - end)
            end = time.time()

        print('[Validation Epoch: {}], Loss: {:.6f}, Acc: {:.5f}%, ACC:{:.5f}%'.format(epoch, val_loss, acc / len(
            val_loader.dataset) * 100., 100. * acc / total_num))

        valid_loss.append(criterion(output, target).item())
        valid_acc_list.append(100. * test_acc)

    return acc / len(val_loader.dataset) * 100.


def save_checkpoint(state, is_best, filename, best_filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)





if __name__ == '__main__':
    main()



