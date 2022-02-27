#print ("Bakugo dies!")
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import os
import sys
import time
from multiprocessing import freeze_support, Process
from IPython.display import HTML
seed = 3143

random.seed(seed)
torch.manual_seed(seed)
dataRoot = "C:\\Users\\wkamf\\Anime GAN\\Data\\"
batchSize = 128
workers = 2
imageSize = 64
nc = 3
nz = 100
ngf = 64
ndf = 64
ne = 100
lr = 0.0002
b1 = 0.5
ngpu = 1

def main():
    #print("Seed is ", seed)
    dataSet = dset.ImageFolder(root = dataRoot,
                               transform = transforms.Compose([
                                   transforms.Resize(imageSize),
                                   transforms.CenterCrop(imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ]))
    dataLoader = torch.utils.data.DataLoader(dataSet, batch_size = batchSize, shuffle = True, num_workers = workers)

    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    realBatch = next(iter(dataLoader))
    plt.figure(figsize = (8, 8))
    plt.axis("off")
    plt.title("Bakugo's Training Images")
    #print(np.abs(np.transpose(vutils.make_grid(realBatch[0].to(device)[:64], padding = 2, Normalize = True).cpu(),(1,2,0))))
    plt.imshow(np.abs(np.transpose(vutils.make_grid(realBatch[0].to(device)[:64], padding = 2, Normalize = True).cpu(),(1,2,0))))
    plt.show()
    #time.sleep(20)
    netG = Generator(ngpu).to(device)
    netG.apply(WeightsInit)
    #print(netG)
    netD = Discriminator(ngpu).to(device)
    netD.apply(WeightsInit)
    #print(netD)
    cLoss = nn.BCELoss()
    fixedNoise = torch.randn(64, nz, 1, 1, device = device)
    realLabel = 1
    fakeLabel = 0
    optimizerD = optim.Adam(netD.parameters(), lr = lr, betas = (b1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr = lr, betas = (b1, 0.999))
    imageList = []
    gLosses = []
    dLosses = []
    iters = 0
    for epoch in range(ne):
        for i, data in enumerate(dataLoader, 0):
            netD.zero_grad()
            realCpu = data[0].to(device)
            bSize = realCpu.size(0)
            label = torch.full((bSize,), realLabel, device = device).to(torch.float32)
            output = netD(realCpu).view(-1).to(torch.float32)
            errorRealD = cLoss(output, label)
            errorRealD.backward()
            dx = output.mean().item()
            noise = torch.randn(bSize, nz, 1, 1, device = device)
            fake = netG(noise)
            label.fill_(fakeLabel)
            output = netD(fake.detach()).view(-1).to(torch.float32)
            errorFakeD = cLoss(output, label)
            errorFakeD.backward()
            DGz1 = output.mean().item()
            errorD = errorRealD + errorFakeD
            optimizerD.step()
            netG.zero_grad()
            label.fill_(realLabel)
            output = netD(fake).view(-1).to(torch.float32)
            errorG = cLoss(output, label)
            errorG.backward()
            DGz2 = output.mean().item()
            optimizerG.step()
            if i % 497 == 0 :
                #print("epoch = ", epoch, "number of epochs = ", ne, "i = ", i, "dataloader length = ", len(dataLoader), "errorD.item() = ", errorD.item(), "errorG.item() = ", errorG.item(), "dx = ", dx, "DGz1 = ", DGz1, "DGz2 = ", DGz2)
                print("(", epoch, " / ", ne, ") i = ", i, "dataloader length = ", len(dataLoader),
                      "errorD.item() = ", errorD.item(), "errorG.item() = ", errorG.item(), "dx = ", dx, "DGz1 = ",
                      DGz1, "DGz2 = ", DGz2)
                #plt.figure(figsize = (16, 9))
                #plt.title("D and G loss training and also death of Bakugo")
                #plt.plot(gLosses, label = "G")
                #plt.plot(dLosses, label = "D")
                #plt.xlabel("iterations")
                #plt.ylabel("loss")
                #plt.legend()
                #plt.show()
                #fig = plt.figure(figsize = (8, 8))
                #im = [[plt.imshow(np.transpose(j, (1, 2, 0)), animated = True)] for j in imageList]
                #anime = animation.ArtistAnimation(fig, im, interval = 1000, repeat_delay = 1000, blit = True)
                #HTML(anime.to_jshtml())
            gLosses.append(errorG.item())
            dLosses.append(errorD.item())
            if (iters % 200 == 0) or ((epoch == ne - 1) and (i == len(dataLoader) -1)):
                with torch.no_grad():
                    fake = netG(noise).detach().cpu()
                imageList.append(vutils.make_grid(fake, padding = 2, normalize = True))
            iters += 1
    plt.figure(figsize=(16, 9))
    plt.title("D and G loss training and also death of Bakugo")
    plt.plot(gLosses, label="G")
    plt.plot(dLosses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.legend()
    plt.show()
    fig = plt.figure(figsize=(8, 8))
    im = [[plt.imshow(np.transpose(j, (1, 2, 0)), animated=True)] for j in imageList]
    anime = animation.ArtistAnimation(fig, im, interval=1000, repeat_delay=1000, blit=True)
    HTML(anime.to_jshtml())
    plt.show()


if __name__ == '__main__':
    freeze_support()
    Process(target = main).start()
def DisplayAnime(_realBatch, _imageList):
    plt.figure(20, 20)
    plt.subplot(1, 2, 1)
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(_realBatch[0].to(device)[:64], padding = 5, normalize = True).cpu(), (1, 2, 0)))

    plt.subplot(1, 2, 2)
    plt.title("Fake Images")
    plt.imshow(np.transpose(_imageList[-1], (1, 2, 0)))
    plt.show()
def WeightsInit(m):
    className = m.__class__.__name__
    if className.find('Conv') != -1 :
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif className.find('BatchNorm') != -1 :
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module) :
    def __init__(self, ngpu) :
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
        nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias = False),
        nn.BatchNorm2d(ngf * 8),
        nn.ReLU(True),
        nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias = False),
        nn.BatchNorm2d(ngf * 4),
        nn.ReLU(True),
        nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias = False),
        nn.BatchNorm2d(ngf * 2),
        nn.ReLU(True),
        nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias = False),
        nn.BatchNorm2d(ngf * 1),
        nn.ReLU(True),
        nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias = False),
        nn.Tanh()
        )
    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias = False),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias = False),
            nn.Sigmoid()
        )
    def forward(self, input):
        return self.main(input)
