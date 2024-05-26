import time

import torch
import torch.utils.data as tud
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms.functional as tvf
import torchvision.datasets as tds
import torchvision.utils as tu


dataset_root = "./"
cpu_num = 4

mnist_train = tds.FashionMNIST(
    root=dataset_root,
    download=True,
    train=True,
    transform=tvf.pil_to_tensor,
)

mnist_eval = tds.FashionMNIST(
    root=dataset_root,
    download=True,
    train=False,
    transform=tvf.pil_to_tensor,
)

batchsize = 32

train_loader = tud.DataLoader(mnist_train, batch_size=batchsize, num_workers=cpu_num, shuffle=True)
val_loader = tud.DataLoader(mnist_eval, batch_size=batchsize, shuffle=True)

class MyCnn(nn.Module):
    def __init__(self, out_sz, ch=32):
        super().__init__()

        self.channels = ch
        self.out_sz = out_sz
        # (28x28)
        self.conv1_1 = nn.Conv2d(1, ch, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(ch, ch*2, kernel_size=3, stride=1, padding=1)

        # (14x14)
        self.lin1 = nn.Linear(ch*2 * 14 * 14, 128, bias=True)
        self.lin2 = nn.Linear(128, out_sz, bias=True)

    def forward(self, x):
        x = self.conv1_1(x)
        x = F.relu(x)
        x = self.conv1_2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        x = x.view(-1, self.channels*2 * 14 * 14)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        return x

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = MyCnn(out_sz=10, ch=16).to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
lossfn = nn.CrossEntropyLoss()
epochs = 10

for epoch in range(epochs):
    t0 = time.time()
    for i, (images, target) in enumerate(train_loader):
        optimizer.zero_grad()
        images = images.float().to(device)
        targets = target.to(device)

        outs = model(images)
        loss = lossfn(outs, targets)
        loss.backward()
        optimizer.step()

    losses = []
    for i, (images, target) in enumerate(val_loader):
        with torch.no_grad():
            images = images.float().to(device)
            targets = target.to(device)
            outs = model(images)
            loss = lossfn(outs, targets)
            losses.append(loss)
    epoch_loss = torch.Tensor(losses).mean().item()
    elapsed = time.time() - t0
    print("Epoch {}: Loss = {}, Elapsed = {:.2f}s".format(epoch, epoch_loss, elapsed))

total = len(mnist_eval)
correct = 0
with torch.no_grad():
    for image, target in mnist_eval:
        pred = model(image.float().to(device))
        pred = F.softmax(pred, dim=1).argmax()
        if pred == target:
            correct += 1

    print("{:.2f}% correct".format(100*correct/total))

try:
    onnx_prog = torch.onnx.export(
        model,
        torch.randn(1, 1, 28, 28).to(device),
        "fmnist.onnx",
        input_names = ['input'],
        output_names = ['output']
    )
except ImportError as e:
    print("Couldn't export to onnx. Are onnx and onnxruntime installed?")
    print(f"{e}")
