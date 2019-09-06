from DataProvider import DataProvider
from Net import Net

import torch
import torch.optim as optim
import torch.nn as nn

import argparse

# ------------------------------- Learning Rate -------------------------------


def adjust_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        if 'lr' in param_group.keys():
            param_group['lr'] = lr


def adjust_lr_by_epoch(optimizer, epoch, base_lr):
    if epoch <= 40:
        adjust_lr(optimizer, base_lr)
    elif epoch <= 80:
        adjust_lr(optimizer, 0.5 * base_lr)
    else:
        adjust_lr(optimizer, 0.1 * base_lr)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        if 'lr' in param_group.keys():
            return param_group['lr']


# --------------------------------- Test Acc ----------------------------------


def test_acc(net):
    with torch.no_grad():
        test_pred_prob = net.forward(test_imgs)
        test_pred = test_pred_prob.max(1).indices
        correct = (test_pred == test_labels).sum().item()
        acc = correct * 100.0 / len(test_imgs)
        return acc


# --------------------------------- Arguments ---------------------------------

WATCH_LOSS_PER_BATCH = 20
SAVE_PER_EPOCH = 5

parser = argparse.ArgumentParser()
parser.add_argument("--train")
parser.add_argument("--test")
parser.add_argument("--init_lr", type=float)
parser.add_argument("--batch_size", type=int)
args = parser.parse_args()

# ------------------------------ data & testdata ------------------------------

provider = DataProvider(args.batch_size, args.train, args.test)

test_imgs = torch.from_numpy(provider.dataset.test_x)
test_imgs = test_imgs.permute(0, 3, 1, 2).float() / 255
test_labels = torch.from_numpy(provider.dataset.test_y)

# ----------------------------- net, opt, lr, loss ----------------------------

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.init_lr, momentum=0.9)

# -------------------------------- train & test -------------------------------

for epoch in range(50):
    sum_loss = 0.0
    adjust_lr_by_epoch(optimizer, epoch, args.init_lr)
    for batch_no in range(1, provider.train_batch_num + 1):
        imgs, labels = provider.next()
        imgs = imgs.permute(0, 3, 1, 2).float() / 255
        pred = net.forward(imgs)
        '''
        相当于loss = criterion.forward(distribution, label_gt)
        注意不要写成 loss = nn.CrossEntropyLoss(distribution, label_gt) !!!
        '''
        loss = criterion(pred, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sum_loss += loss.item()
        if batch_no % WATCH_LOSS_PER_BATCH == 0:
            print("[{}, {}] loss: {:.5f}  lr: {:.5f}".format(
                epoch, batch_no, sum_loss / WATCH_LOSS_PER_BATCH,
                get_lr(optimizer)))
            sum_loss = 0.0

    # this epoch ends. start Testing:
    acc = test_acc(net)
    print("\n[{}] acc in Test: {:.1f}\n".format(epoch, acc))

    # this epoch ends. start Saving Model to ./model/:
    if epoch % SAVE_PER_EPOCH == 0:
        state = {
            'epoch': epoch,
            'acc': acc,
            'net': net.state_dict(),
            'opt': optimizer.state_dict(),
        }
        torch.save(state, 'model/ckpt_{}.mdl'.format(epoch))

# --------------------------------- Load model --------------------------------

checkpoint = torch.load('model/ckpt_35.mdl')
net.load_state_dict(checkpoint['net'])
optimizer.load_state_dict(checkpoint['opt'])
epoch = checkpoint['epoch']
print("Reading epoch {}, acc is: {}".format(epoch, test_acc(net)))
