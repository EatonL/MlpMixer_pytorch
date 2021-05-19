import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from tqdm import tqdm
from model import Mlp_Mixer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device is " + str(device))

@torch.no_grad()
def compute_acc(model, test_dataloader):
    right_num = 0
    total_num = 0

    for in_data, label in tqdm(test_dataloader):
        in_data = in_data.to(device)
        label = label.to(device)
        total_num += len(in_data)
        out = model(in_data)
        pred = out.argmax(dim=-1)
        right_num += (pred == label).sum()
    
    print('Accuracy: {}'.format(right_num.item() / total_num))
    return (right_num.item() / total_num)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=64, help='batch size for dataloader')
    # parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', type=str, default=None, help='resume training')
    parser.add_argument('-work_dir', type=str, default='./test', help='folder to save the result')
    args = parser.parse_args()

    best_acc = 0.0
    max_epoch = 100
    all_iter = 0
    train_all_time = 0
    save_epoch = 10
    lr = args.lr
    batch_size = args.b

    train_writer = SummaryWriter(log_dir=os.path.join(args.work_dir,'tf_logs'))
    model = Mlp_Mixer(dim=3, inter_dim=1024,
                      token_inter_dim=512, channel_inter_dim=4096,
                      img_size=(224, 224), patch_size=(16, 16),
                      num_block=24, num_class=100, dropout_ratio=0
                      )
    print(model)
    model = model.to(device)

    transform = transforms.Compose([transforms.Resize(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])

    data_train = datasets.CIFAR100(root="./data/",
                                   transform=transform,
                                   train=True)
    data_val = datasets.CIFAR100(root="./data/",
                                 transform=transform,
                                 train=False)

    data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                    batch_size=batch_size,
                                                    num_workers=4,
                                                    shuffle=True)
    data_loader_val = torch.utils.data.DataLoader(dataset=data_val,
                                                  batch_size=batch_size,
                                                  num_workers=4,
                                                  shuffle=True)


    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()
    train_num_iter = len(data_loader_train)
    # train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    # warmup_scheduler = WarmUpLR(optimizer, train_num_iter * args.warm)

    if args.resume:
        print('\n--- loading checkpoint: {} ---'.format(args.resume))
        model.load_state_dict(torch.load(args.resume))
    
    for epoch in range(1,max_epoch+1):
        model.train(mode=True)
        iter = 0
        loss_epoch = 0
        
        # if epoch <= args.warm:
        #     warmup_scheduler.step()
            
        with tqdm(total=train_num_iter) as train_bar:
            for iter, data in enumerate(data_loader_train):

                train_start = time.time()
                date_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

                input = data[0].to(device)
                label = data[1].to(device)

                output = model(input)
                pred = output.argmax(dim=-1)
                corr = (pred == label).sum()

                loss = loss_func(output, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_epoch += loss.item()

                train_writer.add_scalar('Train/Loss', loss.item(), global_step=all_iter)
                train_writer.add_scalar('Train/Accuracy', corr.item()/len(input), global_step=all_iter)

                train_end = time.time()
                train_iter_time = train_end - train_start
                train_all_time += train_iter_time

                train_bar.set_description('{}. Epoch: {}/{}, Iter: {}/{}, time: {:.3f},lr: {:.5f} Loss: {:.5f}, Accuracy: {:.2f}'
                                            .format(date_time, epoch, max_epoch, iter, train_num_iter,
                                                    train_all_time,lr,
                                                    loss_epoch / train_num_iter, corr.item() / len(input)))
                train_all_time = 0
                iter += 1
                all_iter += 1
                train_bar.update(1)

        if not os.path.exists(args.work_dir):
            os.mkdir(args.work_dir)
            
        if epoch % save_epoch == 0:
            print('\n--- Saving checkpoint at epoch {} ---'.format(epoch + 1))
            torch.save(model.state_dict(), os.path.join(args.work_dir,"epoch_{}.pth".format(epoch)))
        
        acc_val = compute_acc(model, data_loader_val)
        train_writer.add_scalar('Val/Accuracy', acc_val, global_step=epoch)
        
    train_writer.close()
