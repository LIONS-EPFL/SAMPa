import argparse

import torch
import torch.optim as optim
from lightning.fabric import Fabric, seed_everything
import os

from get_dataset import get_dataset
from get_model import get_model
from utility import smooth_crossentropy, unflatten

    
    
def run(args):        
    fabric = Fabric(devices=2, strategy="ddp", accelerator="cuda")
    fabric.launch()
    
    if fabric.global_rank == 0:
        save_path = args.save_path
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
            
    seed_everything(args.seed)
    
    dataset = get_dataset(args)

    train_loader, valid_loader, test_loader = fabric.setup_dataloaders(dataset.train, dataset.valid, dataset.test, use_distributed_sampler=False)

    model = get_model(args) 
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

    model, optimizer = fabric.setup(model, optimizer)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    y_grads = torch.cat([torch.flatten(torch.zeros_like(param.data)) for param in model.parameters()])
    pert_grads, agg_grads = torch.zeros_like(y_grads), torch.zeros_like(y_grads)
    
    data1, target1 = None, None
    best_acc = 0

    for epoch in range(1, args.epochs + 1):
        # TRAINING LOOP
        model.train()
        lr = optimizer.param_groups[0]['lr']
        
        for batch_idx, (data, target) in enumerate(train_loader):
            with fabric.no_backward_sync(model, enabled=True):
                if data1 is None:
                    data1, target1 = data, target

                if fabric.global_rank == 0:
                    y_grads = (agg_grads - pert_grads)/(args.og+1e-15)
                    
                    scale = args.rho / (y_grads.norm(p=2) + 1e-12)
                    y_grads *= -scale   # Notice '-' here, just for the consistency with descent
                    y_grads_list = unflatten(y_grads, model.parameters())
                    for p, y_g in zip(model.parameters(), y_grads_list):
                        p.data.sub_(y_g)
                    output1 = model(data1)
                    loss = smooth_crossentropy(output1, target1, smoothing=0.1)
                    fabric.backward(loss.mean()) 

                    pert_grads = torch.cat([param.grad.detach().view(-1) for param in model.parameters()]) * (1-args.og)
                    agg_grads.copy_(pert_grads)
                    
                elif fabric.global_rank == 1:
                    monmentum = []
                    for group in optimizer.param_groups:
                        for p in group["params"]:
                            # Momentum
                            if isinstance(optimizer, torch.optim.SGD):
                                if p in optimizer.state and 'momentum_buffer' in optimizer.state[p]:
                                    if optimizer.state[p]["momentum_buffer"] is not None:
                                        grad = group['weight_decay'] * p.data      # weight_decay
                                        grad += optimizer.state[p]["momentum_buffer"] * group["momentum"]
                                        monmentum.append(torch.flatten(grad))

                    y_grads = lr * (y_grads + torch.cat(monmentum)) if monmentum else lr * y_grads
                    y_grads_list = unflatten(y_grads, model.parameters())
                    for p, y_g in zip(model.parameters(), y_grads_list):
                        p.data.sub_(y_g)
                    output = model(data)
                    loss = smooth_crossentropy(output, target, smoothing=0.1).mean()
                    fabric.backward(loss) 
                    
                    y_grads = torch.cat([param.grad.detach().view(-1) for param in model.parameters()])
                    agg_grads.copy_(args.og * y_grads)
                    
                agg_grads = fabric.all_reduce(agg_grads, reduce_op='sum')
                
                # Recover the weights of model and set the gradients
                begin = 0
                for p, y_g in zip(model.parameters(), y_grads_list):
                    p.data.add_(y_g)
                    size = p.view(-1).shape[0]
                    p.grad = agg_grads[begin:begin+size].view(p.shape)
                    begin += size
                
                optimizer.step()
                optimizer.zero_grad()
                        
                data1, target1 = data, target
            
        scheduler.step()

        # Validate
        if fabric.global_rank == 0:
            model.eval()
            loss, correct, steps = 0, 0, 0
            with torch.no_grad():
                for data, target in valid_loader:
                    predictions = model(data) 
                    loss += smooth_crossentropy(predictions, target).sum().item()
                    correct += (torch.argmax(predictions, 1) == target).sum().item()
                    steps += len(target)
            loss = loss/steps
            acc = correct/steps
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), save_path+"/best_model.pth")
                
            print(f"Epoch {epoch}: Validate loss: {loss:.4f}, Validate accuracy: {100 * acc:.2f}%\n")

    # Test
    if fabric.global_rank == 0:
        model.load_state_dict(torch.load(save_path+"/best_model.pth"))
        model.eval()
        total_loss, correct, steps = 0, 0, 0

        with torch.no_grad():
            for data, target in test_loader:
                predictions = model(data)
                loss = smooth_crossentropy(predictions, target)
                total_loss += loss.sum().item()
                correct += (torch.argmax(predictions, 1) == target).sum().item()
                steps += len(target)   
            loss = total_loss/steps
            acc = correct/steps

            print(f"Best_Test_Accuracy: {100 * acc:.2f}%\n")
            print(f"\Best_Test_Loss: {loss:.4f}\n")

        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAMPa in parallel")
    parser.add_argument(
        "--batch-size", type=int, default=128, metavar="N", help="input batch size for training"
    )
    parser.add_argument("--epochs", type=int, default=5, metavar="N", help="number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.1, metavar="LR", help="learning rate")
    parser.add_argument("--rho", default=0.1, type=float, help="\rho parameter for SAMPa.")
    parser.add_argument("--og", default=0.2, type=float, help="\lambda for SAMPa.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--seed", type=int, default=42, metavar="S", help="random seed")
    parser.add_argument("--save_path", type=str, default='./save/', help="The path to save the model.")
    parser.add_argument("--dataset", type=str, default='cifar10')
    parser.add_argument("--model", type=str, default='resnet56')
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    run(args)