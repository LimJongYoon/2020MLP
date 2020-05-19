#%matplotlib inline

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchsummary import summary
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor(), #numpy to tensor
    transforms.Normalize((0.5, ), (0.5, )),
])

batch_size = 128 #배치사이즈
z_dim = 100 #노드

database = dataset.MNIST('mnist', train = True, download = True, transform = transform)
#Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
train_loader = torch.utils.data.DataLoader(
    #dataset.MNIST('mnist', train = True, download = True, transform = transform),
    database,
    batch_size = batch_size,
    shuffle = True
)
for i, data in enumerate(train_loader):
     print ("batch id =" + str(i) )
     print (data[0])
     print (data[1])


def weights_init(m): #클래스 이름별로 분류해서 값 설정 
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Generator_model(nn.Module): #생성 학습모델 
    def __init__(self, z_dim):
        super().__init__()
        self.fc = nn.Linear(z_dim, 256 * 7 * 7)
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), #입력이미지에 2d conv 연산자 적용 
            nn.BatchNorm2d(128), #4차원입력을 정규화 이게왜 필요한지..?
            nn.LeakyReLU(0.01), #특정값 이상의 값만 받음 
            nn.ConvTranspose2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),
            nn.Tanh()
        )
    def forward(self, input):
        x = self.fc(input)
        x = x.view(-1, 256, 7, 7) #텐서 모양 변경 
        return self.gen(x)

generator = Generator_model(z_dim).to(device)
generator.apply(weights_init)
summary(generator, (100, ))

class Discriminator_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01)
        )
        self.fc = nn.Linear(2048, 1)
    def forward(self, input):
        x = self.disc(input)
        return F.sigmoid(self.fc(x.view(-1, 2048)))


discriminator = Discriminator_model().to(device)
discriminator.apply(weights_init) #가중치 
summary(discriminator, (1, 28, 28))
criterion = nn.BCELoss() #손실함수 
fixed_noise = torch.randn(64, z_dim, device=device) #정규분포 
doptimizer = optim.Adam(discriminator.parameters())
goptimizer = optim.Adam(generator.parameters())
real_label, fake_label = 1, 0
image_list = []
g_losses = []
d_losses = []
iterations = 0
num_epochs = 50

for epoch in range(num_epochs):
    print(f'Epoch : | {epoch + 1:03} / {num_epochs:03} |')
    for i, data in enumerate(train_loader):

        discriminator.zero_grad()

        real_images = data[0].to(device)  # real_images: size = (128,1,28,28)

        size = real_images.size(0)  # size = 128 = batch size
        label = torch.full((size,), real_label, device=device)  # real_label =1
        d_output = discriminator(real_images).view(-1)
        derror_real = criterion(d_output, label)

        derror_real.backward()

        noise = torch.randn(size, z_dim, device=device)  # noise shape = (128, 100)
        fake_images = generator(noise)  # fake_images: shape = (128,1,28,28)
        label.fill_(0)  # _: in-place-operation
        d_output = discriminator(fake_images.detach()).view(-1)

        derror_fake = criterion(d_output, label)
        derror_fake.backward()

        derror_total = derror_real + derror_fake
        doptimizer.step()

        generator.zero_grad()
        # label.fill_(real_images) #_: in-place-operation; the same as label.fill_(1)
        label.fill_(1)  # why is the label for the fake-image is one rather than zero?
        d_output = discriminator(fake_images).view(-1)
        gerror = criterion(d_output, label)
        gerror.backward()

        goptimizer.step()

        if i % 50 == 0:  # for every 50th i 50개마다 
            print(
                f'| {i:03} / {len(train_loader):03} | G Loss: {gerror.item():.3f} | D Loss: {derror_total.item():.3f} |')
            g_losses.append(gerror.item())
            d_losses.append(derror_total.item())

        if (iterations % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(train_loader) - 1)): #반복 
            with torch.no_grad():  # check if the generator has been improved from the same fixed_noise vector
                fake_images = generator(fixed_noise).detach().cpu()
            image_list.append(vutils.make_grid(fake_images, padding=2, normalize=True))

        iterations += 1


plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(g_losses,label="Generator")
plt.plot(d_losses,label="Discriminator")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

for image in image_list:
    plt.imshow(np.transpose(image,(1,2,0)))
    plt.show()
