 
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 14:04:40 2021

@author: ppyt
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 12:17:15 2020

@author: ppyt
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch.nn.functional as F
plt.style.use('default')
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sampler import BalancedBatchSampler
from torch.autograd import Variable
from torchvision import transforms as T
from matplotlib import rcParams
from sklearn.manifold import TSNE    
params={'font.family':'serif',
        'font.serif':'Times New Roman',
        'font.style':'italic',
        'font.weight':'normal', #or 'blod'
        'font.size':'14',#or large,small
        }
rcParams.update(params)
font = {'family': 'serif',
        'style':'italic',
        'color':  'white',
        'weight': 'normal',
        }


def __Gussian_Noise(img):
    """
    :param img: 输入的图像
    :param pos: 图像截取的位置,类型为元组，包含(x, y)
    :param size: 图像截取的大小
    :return: 返回截取后的图像
    """
    m = img.shape
    noise = torch.randn(m)
    return img + 0.001*noise

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [x, self.transform(x)]


class MyDataset(Dataset):
    def __init__(self, x_dir, y_dir, transform=None):
        self.X = np.load(x_dir)
        self.X = torch.from_numpy(self.X)
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = np.load(y_dir)
        self.y = torch.from_numpy(self.y)
        self.y = torch.tensor(self.y, dtype=torch.long)
        self.transform = transform
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x = self.X[idx]
        label = self.y[idx]
        if self.transform:
            x = self.transform(x)
        return x,label
class SparseAutoencoder(nn.Module):
  def __init__(self):
    super(SparseAutoencoder, self).__init__()
    self.encoder = nn.Sequential(
        nn.Linear(52, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.ReLU()
        )
    self.decoder = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 52),
        nn.Sigmoid()
        
        )
  
  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x

def add_noise(img):
    noise = torch.randn(img.size()) * 0.2
    noisy_img = img + noise
    return noisy_img

def kl_divergence(rho, rho_hat):
    rho_hat = torch.mean(F.sigmoid(rho_hat), 1) # sigmoid because we need the probability distributions
    rho = torch.tensor([rho] * len(rho_hat))
    return torch.sum(rho * torch.log(rho/rho_hat) + (1 - rho) * torch.log((1 - rho)/(1 - rho_hat)))
# define the sparse loss function
def sparse_loss(rho, images):
    values = images
    loss = 0
    for  i in range(2):
        fc_layer = list(net.encoder.children())[2 * i]
        values = fc_layer(values)
        loss += kl_divergence(rho, values)
    for  i in range(1):
        fc_layer = list(net.decoder.children())[2 * i]
        values = fc_layer(values)
        loss += kl_divergence(rho, values)
    return loss


        


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07*0.5, contrast_mode='one',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
      
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
    

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T),self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask),1,torch.arange(batch_size * anchor_count).view(-1, 1),0)
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        logits = logits * logits_mask 
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
#        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1)+1)
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

#        
def train_autoencoder(net):
    epochs = 2000
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        runningloss = 0
        for images, labels in trainloader_AE:
            optimizer.zero_grad()
           
            output = net(images)
            mse_loss = criterion(output, images)
           
            # add the sparsity penalty
            loss = mse_loss
            loss.backward()
            optimizer.step()
            runningloss += loss.item()/images.shape[0]
#        print('Epoch: {}/{} \t Mean Square Error Loss: {}'.format(epoch+1, epochs, runningloss))
    return net


class classfier(nn.Module):
  def __init__(self,net):
    super(classfier, self).__init__()
    self.encoder = nn.Sequential(*list(net.children())[:-1])
    self.fc = nn.Linear(10, 10)
  
  def forward(self, x):
    x = self.encoder(x)
    x = self.fc(x)
    
    return x

def train_dnn(net):

    net = classfier(net)
    criterion2 =  SupConLoss()
    optimizer = optim.Adam(net.parameters(), lr = 0.001)
    epochs = 5000
    for epoch in range(epochs):
      runningloss = 0
      for images, labels in trainloader_SCL:
        images = torch.cat([images[0], images[1]], dim=0)
        bsz = labels.shape[0]
        features = net.encoder(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        features = F.normalize(features, dim=2)
        optimizer.zero_grad()
        loss2 = criterion2(features,labels)
        loss = loss2
        loss.backward()
        optimizer.step()
        runningloss += loss.item()/images.shape[0]
        net.eval()
#        print(runningloss)

    return net

if __name__ == "__main__":
    BETA = 0
    RHO = 0.5
    transform = transforms.ToTensor()
    X_train = 'X_train_multiclass_shot.npy'
    y_train = 'y_train_multiclass_shot.npy'
    X_test = 'X_test_multiclass_shot.npy'
    y_test = 'y_test_multiclass_shot.npy'
    train_transform = T.Lambda(lambda img:__Gussian_Noise(img))
    data_train_AE =  MyDataset(X_train, y_train)
    data_train_SCL =  MyDataset(X_train, y_train,transform=TwoCropTransform(train_transform))
    data_test = MyDataset(X_test, y_test)
    trainloader_AE = torch.utils.data.DataLoader(data_train_AE, batch_size=128)
    trainloader_SCL = torch.utils.data.DataLoader(data_train_SCL, sampler=BalancedBatchSampler(data_train_SCL.X,data_train_SCL.y),batch_size=70)
    testloader = torch.utils.data.DataLoader(dataset=data_test,batch_size=128,shuffle=True)
    net = SparseAutoencoder()
    optimizer = optim.Adam(net.parameters(), lr = 0.001)
    net = train_autoencoder(net)
    net = train_dnn(net)
    torch.save(net, './AE_Multiclass_shot.pkl')
    X_train = np.load('X_train_multiclass_shot.npy')
    y_train = np.load('y_train_multiclass_shot.npy')
#    X_train = X_train[400:,:]
    X_train = torch.from_numpy(X_train).type(torch.FloatTensor)
    data = net.encoder(X_train)
    data = data.detach().numpy()
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    data = tsne.fit_transform(data)
    for label in range(10):
#        idx = (y_train[400:]== label)
        idx = (y_train == label)
        plt.scatter(data[idx, 0], data[idx, 1])
    plt.legend(['Normal']+['Fault '+ str(x) for x in ['1','2','4','6','7','8','12','14','18']])
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    
    plt.savefig('./Long_tail_Tsne_shot.png',dpi=600)


#  
#plt.legend(['Normal']+['Fault '+ str(x) for x in ['1','2','3','8','10','11','12','13','14','20']])
#plt.xlabel('Dim 1')
#plt.ylabel('Dim 2')
#plt.show()

