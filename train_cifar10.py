from lib.ModelWrapper import ModelWrapper
from tensorboardX import SummaryWriter
import torch
from torchvision import transforms, datasets
from archs.cifar10 import vgg, resnet
import numpy as np
import random
import os

data_name = 'cifar10'
model_name = 'resnet'

# setting
lr = 1e-4
train_batch_size = 128
train_epoch = 4000
eval_batch_size = 256
label_noise = 0.15
k = 64


dataset = datasets.CIFAR10
train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor()])

eval_transform = transforms.Compose([transforms.ToTensor()])

if model_name == 'vgg16':
    model = vgg.vgg16_bn()
elif model_name == 'resnet':
    model = resnet.resnet18(k)
else:
    raise Exception("No such model!")

# load data
train_data = dataset('D:/Datasets', train=True, transform=train_transform)
train_targets = np.array(train_data.targets)
data_size = len(train_targets)
random_index = random.sample(range(data_size), int(data_size*label_noise))
random_part = train_targets[random_index]
np.random.shuffle(random_part)
train_targets[random_index] = random_part
train_data.targets = train_targets.tolist()

noise_data = dataset('D:/Datasets', train=True, transform=train_transform)
noise_data.targets = random_part.tolist()
noise_data.data = train_data.data[random_index]


test_data = dataset('D:/Datasets', train=False, transform=eval_transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=True, num_workers=0,
                                           drop_last=False)
noise_loader = torch.utils.data.DataLoader(noise_data, batch_size=train_batch_size, shuffle=False, num_workers=0,
                                           drop_last=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=eval_batch_size, shuffle=False, num_workers=0,
                                          drop_last=False)

# build model
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
wrapper = ModelWrapper(model, optimizer, criterion, device)

# train the model
save_path = os.path.join('runs', data_name, "{}_{}_k{}".format(model_name, int(label_noise*100), k))
if not os.path.exists(save_path):
    os.makedirs(save_path)
np.savez(os.path.join(save_path, "label_noise.npz"), index=random_index, value=random_part)
writer = SummaryWriter(logdir=os.path.join(save_path, "log"))

itr_index = 1
wrapper.train()

for id_epoch in range(train_epoch):
    # train loop

    for id_batch, (inputs, targets) in enumerate(train_loader):

        loss, acc, _ = wrapper.train_on_batch(inputs, targets)
        print("epoch:{}/{}, batch:{}/{}, loss={}, acc={}".
              format(id_epoch+1, train_epoch, id_batch+1, len(train_loader), loss, acc))
        if itr_index % 20 == 0:
            writer.add_scalar("train acc", acc, itr_index)
            writer.add_scalar("train loss", loss, itr_index)

        itr_index += 1

    wrapper.eval()
    test_loss, test_acc = wrapper.eval_all(test_loader)
    noise_loss, noise_acc = wrapper.eval_all(noise_loader)
    print("epoch:{}/{}, batch:{}/{}, testing...".format(id_epoch + 1, train_epoch, id_batch + 1, len(train_loader)))
    print("clean: loss={}, acc={}".format(test_loss, test_acc))
    print("noise: loss={}, acc={}".format(noise_loss, noise_acc))
    print()
    writer.add_scalar("test acc", test_acc, itr_index)
    writer.add_scalar("test loss", test_loss, itr_index)
    writer.add_scalar("noise acc", noise_acc, itr_index)
    writer.add_scalar("noise loss", noise_loss, itr_index)
    state = {
        'net': model.state_dict(),
        'optim': optimizer.state_dict(),
        'acc': test_acc,
        'epoch': id_epoch,
        'itr': itr_index
    }
    torch.save(state, os.path.join(save_path, "ckpt.pkl"))
    writer.flush()
    # return to train state.
    wrapper.train()

writer.close()
