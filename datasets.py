'''
define the dataset & set up data
'''

import torch
import math
import torchvision.datasets as datasets
import os
import torchvision.transforms as transforms
import PIL



def load_checkpoint(model, ckpt_path):
    checkpoint = torch.load(ckpt_path)
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    ckpt = {}
    for k, v in checkpoint.items():
        if k.startswith('module.'):
            ckpt[k[7:]] = v
        else:
            ckpt[k] = v
    model.load_state_dict(ckpt)




def get_train_dataset(args, trans):
    image_path = "data"
    train_dataset = datasets.ImageFolder(root=image_path + "/train",
                                             transform=trans)
    return train_dataset


def get_val_dataset(args, trans):
    image_path = "data"
    val_dataset = datasets.ImageFolder(root=image_path + "/test",
                                           transform=trans)
    return val_dataset


def get_test_dataset(args, trans):
    image_path = "data"
    test_dataset = datasets.ImageFolder(root=image_path + "/test",
                                           transform=trans)
    return test_dataset



    
def get_default_train_trans(args):
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],          
                                     std=[0.5, 0.5, 0.5])
    if (not hasattr(args, 'resolution')) or args.resolution == 224:
        trans = transforms.Compose([
            transforms.RandomResizedCrop((128, 128)),        
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
    else:
        raise ValueError('Not yet implemented.')
    return trans


def get_default_val_trans(args):
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],          
                                     std=[0.5, 0.5, 0.5])
    if (not hasattr(args, 'resolution')) or args.resolution == 224:
        trans = transforms.Compose([
            transforms.Resize((128, 128)),  
            transforms.ToTensor(),
            normalize])
    else:
        raise ValueError('Not yet implemented.')
    return trans


def get_default_test_trans(args):
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],        
                                     std=[0.5, 0.5, 0.5])
    if (not hasattr(args, 'resolution')) or args.resolution == 224:
        trans = transforms.Compose([
            transforms.Resize((128, 128)),  
            transforms.ToTensor(),
            normalize])
    else:
        trans = transforms.Compose([
            transforms.Resize(args.resolution, interpolation=PIL.Image.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            normalize,
        ])
    return trans




def get_default_val_loader(args):
    val_trans = get_default_val_trans(args)
    val_dataset = get_val_dataset(args, val_trans)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.val_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    return val_loader


def get_default_test_loader(args):
    test_trans = get_default_test_trans(args)
    test_dataset = get_test_dataset(args, test_trans)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.val_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    return test_loader
    
    
    
    
def get_default_train_sampler_loader(args):
    train_trans = get_default_train_trans(args)
    train_dataset = get_train_dataset(args, train_trans)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    return train_sampler, train_loader



