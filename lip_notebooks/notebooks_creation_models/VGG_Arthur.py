from deel import torchlip
import torch
import torch.nn as nn
from orthogonium.layers import BatchCentering2D
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

class HKRMultiLossLSE(torch.nn.Module):
    """
    Loss that estimates the Wasserstein-1 distance using the Kantorovich-Rubinstein
    duality with a hinge regularization and logsumexp summary.
    """

    def __init__(
        self,
        alpha: float = 1.,
        temperature: float = 1.,
        penalty = 1., # max <logsumpexp< max+penalty*margin
        margin = 1,
    ):
        """
        Args:
            alpha: Regularization factor between the hinge and the KR loss.
            min_margin: Minimal margin for the hinge loss.
            true_values: tuple containing the two label for each predicted class.
        """
        super().__init__()
        self.penalty = penalty
        self.alpha = alpha
        self.temperature = temperature
        self.margin = margin


    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        #temperature scaling -> let the margin fixed
        y_pred = y_pred * self.temperature
        
        #positive examples
        pos = y_pred[y_true == 1]
        hinge_pos = torch.mean(F.relu(self.margin -  pos))
        kr_pos = torch.mean(pos)


        #negative examples -> summarized to one value with logsumexp
        neg = torch.where(y_true == 1, -float('inf'),  y_pred)
        nb_bins = y_pred.new_tensor(y_pred.size(1) - 1)
        nb_bins = torch.log(nb_bins)
        t = nb_bins/ (self.margin*self.penalty)
        neg_soft = 1/t*torch.logsumexp(t*neg,dim = 1)# max <neg_soft< max+penalty*margin
        
        hinge_neg = torch.mean(F.relu(self.margin +  neg_soft))
        kr_neg = torch.mean(neg_soft)
      
        hinge_loss = hinge_pos + hinge_neg
        kr = kr_pos  - kr_neg
        loss_val = (1-1./self.alpha)*hinge_loss -1./self.alpha* kr
        return loss_val
    
class MultiplyByScalar(nn.Module):
    def __init__(self, coeff):
        super(MultiplyByScalar, self).__init__()
        self.coeff = coeff

    def forward(self, x):
        return x * self.coeff
    
class LipBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation, coeff):
        super(LipBlock, self).__init__()
        self.conv = torchlip.SpectralConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False)
        self.norm = BatchCentering2D(out_channels)
        self.activation = activation
        self.scalar = MultiplyByScalar(coeff)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.scalar(x)
        return x
    
def load_model():
    coeff_total = 500 # Change according to temperature
    nb_blocks = 12
    coeff_block = coeff_total**(1/nb_blocks)
    width = 1.5  # 1: 2.5M parameters, 2: 10M parameters, 4: 40M parameters

    model = torchlip.Sequential(
        LipBlock(3, int(width*64), activation=torchlip.GroupSort2(), coeff=coeff_block),
        LipBlock(int(width*64), int(width*64), activation=torchlip.GroupSort2(), coeff=coeff_block),
        LipBlock(int(width*64), int(width*64), activation=torch.abs, coeff=coeff_block),
        torchlip.ScaledL2NormPool2d(kernel_size=2, stride=2),
        LipBlock(int(width*64), int(width*128), activation=torchlip.GroupSort2(), coeff=coeff_block),
        LipBlock(int(width*128), int(width*128), activation=torchlip.GroupSort2(), coeff=coeff_block),
        LipBlock(int(width*128), int(width*128), activation=torch.abs, coeff=coeff_block),
        torchlip.ScaledL2NormPool2d(kernel_size=2, stride=2),
        LipBlock(int(width*128), int(width*256), activation=torchlip.GroupSort2(), coeff=coeff_block),
        LipBlock(int(width*256), int(width*256), activation=torchlip.GroupSort2(), coeff=coeff_block),
        LipBlock(int(width*256), int(width*256), activation=torch.abs, coeff=coeff_block),
        torchlip.ScaledL2NormPool2d(kernel_size=2, stride=2),
        LipBlock(int(width*256), int(width*512), activation=torchlip.GroupSort2(), coeff=coeff_block),
        LipBlock(int(width*512), int(width*512), activation=torchlip.GroupSort2(), coeff=coeff_block),
        LipBlock(int(width*512), int(width*512), activation=torch.abs, coeff=coeff_block),
        torchlip.ScaledAdaptativeL2NormPool2d((1, 1)),
        nn.Flatten(),
        torchlip.SpectralLinear(int(width*512), 10),
        MultiplyByScalar(1/coeff_total)
    )
    return model