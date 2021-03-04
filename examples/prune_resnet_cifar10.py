import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# from cifar_resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
import cifar_resnet as resnet

import torch_pruning as tp
import argparse
import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn 
import numpy as np 
from ptflops import get_model_complexity_info

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--verbose', action='store_true', default=False)
parser.add_argument('--total_epochs', type=int, default=50)
parser.add_argument('--step_size', type=int, default=70)
parser.add_argument('--prune', type=int, default=5)
parser.add_argument('--epoch_prune', type=int, default=10)

args = parser.parse_args()

def get_dataloader():
    train_loader = torch.utils.data.DataLoader(
        CIFAR10('./data', train=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]), download=True),batch_size=args.batch_size, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        CIFAR10('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
        ]),download=True),batch_size=args.batch_size, num_workers=2)
    return train_loader, test_loader

def eval(model, test_loader):
    correct = 0
    total = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (img, target) in enumerate(test_loader):
            img = img.to(device)
            out = model(img)
            pred = out.max(1)[1].detach().cpu().numpy()
            target = target.cpu().numpy()
            correct += (pred==target).sum()
            total += len(target)
    return correct / total

def train_model(model, model_name, train_loader, test_loader, epochs, prune=False):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, 0.1)
    model.to(device)

    best_acc = -1
    for epoch in range(epochs):
        model.train()
        for i, (img, target) in enumerate(train_loader):
            img, target = img.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(img)
            loss = F.cross_entropy(out, target)
            loss.backward()
            optimizer.step()
            if i%10==0 and args.verbose:
                logs("Epoch %d/%d, iter %d/%d, loss=%.4f"%(epoch, epochs, i, len(train_loader), loss.item()))
        model.eval()
        acc = eval(model, test_loader)
        logs("Epoch %d/%d, Acc=%.4f"%(epoch, epochs, acc))
        if best_acc < acc:
            best_acc = acc
            if not prune:
                torch.save(model, f'Resnet_train_{model_name}.pth')
        scheduler.step()
    logs("Best Acc=%.4f"%(best_acc))
    
    

def prune_model(model, prune_prob = 0.1):
    logs(f"Pruning model with prune_prob = {prune_prob}")
    model.cpu()
    DG = tp.DependencyGraph().build_dependency( model, torch.randn(1, 3, 32, 32) )
    def prune_conv(conv, amount=0.2):
        strategy = tp.strategy.L1Strategy()
        pruning_index = strategy(conv.weight, amount=amount)
        plan = DG.get_pruning_plan(conv, tp.prune_conv, pruning_index)
        plan.exec()
    
    for _, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            prune_conv(m, prune_prob)
    return model  

def logs(str):
    print(str)
    with open("logs.txt", "a") as f:
        f.write(str + '\n')

def main():
    train_loader, test_loader = get_dataloader()
    list_model = ['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152']
    
    for model_name in list_model:
        logs(f"Model name: {model_name}")
        model = getattr(resnet, model_name)(num_classes = 10)
        logs(f"Train model with {args.total_epochs} epochs")
        train_model(model, model_name, train_loader, test_loader, epochs = args.total_epochs)
        macs, params = get_model_complexity_info(model, (3, 32, 32), as_strings = True, print_per_layer_stat = True, verbose = True)
        logs('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        logs('{:<30}  {:<8}'.format('Numper of parameters: ', params))
        for i in range(args.prune):
            if i == 0:
                model = torch.load(f"Resnet_train_{model_name}.pth")
            else:
                model = torch.load(f"Resnet_prune_{model_name}_{i - 1}.pth")
            logs(f"Prune {i}/{args.prune}")
            prune_model(model)
            params = sum([np.prod(p.size()) for p in model.parameters()])
            logs("Number of Parameters: %.1fM"%(params/1e6))
            train_model(model, model_name, train_loader, test_loader, epochs = args.epoch_prune, prune = True)
            macs, params = get_model_complexity_info(model, (3, 32, 32), as_string = True, print_per_layer_stat = True, verbose = True)
            logs('{:<30}  {:<8}'.format('Computational complexity: ', macs))
            logs('{:<30}  {:<8}'.format('Numper of parameters: ', params))
            torch.save(model, f"Resnet_prune_{model_name}_{i}.pth")
        

if __name__=='__main__':
    main()
