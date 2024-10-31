from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={str(self.eps)})"


class IBN(nn.Module):
    r"""Instance-Batch Normalization layer from
    `"Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net"
    <https://arxiv.org/pdf/1807.09441.pdf>`
    Args:
        planes (int): Number of channels for the input tensor
        ratio (float): Ratio of instance normalization in the IBN layer
    """

    def __init__(self, planes, ratio):
        super(IBN, self).__init__()
        self.half = int(planes * ratio)
        self.IN = nn.InstanceNorm2d(self.half, affine=True)
        self.BN = nn.BatchNorm2d(planes - self.half)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out
    

import torch
import torch.nn as nn


class Bottleneck(nn.Module):

    expansion: int = 4

    def __init__(
        self, in_channels: int, out_channels: int, last: bool = False, downsample=None, stride=1, bias: bool = True
    ):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        if not last:
            # Apply Instance normalization in first half channels (ratio=0.5)
            self.ibn = IBN(out_channels, ratio=0.5)
        else:
            self.ibn = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0, bias=bias
        )
        self.batch_norm3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        residual = x.clone()
        # print('*'*50)
        # print(f"input has shape {x.shape}")

        x = self.conv1(x)
        # print(f"after conv1 x has shape {x.shape}")
        x = self.ibn(x)
        x = self.relu(x)
        # print(f"after ibn and relu x has shape {x.shape}")

        x = self.conv2(x)
        # print(f"after conv2 x has shape {x.shape}")

        x = self.batch_norm2(x)
        x = self.relu(x)

        x = self.conv3(x)
        # print(f"after conv3 x has shape {x.shape}")


        x = self.batch_norm3(x)
        x = self.relu(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out = residual + x
        
        # print(f"out has shape {out.shape}")
        # print('*'*50)


        out = self.relu(out)

        return out


class Resnet50(nn.Module):
    def __init__(
        self,
        ResBlock: Bottleneck,
        emb_dim: int = 2048,
        num_channels: int = 1,
        num_classes: int = 8858,
        dropout=0.1,
        n_bins=84
    ) -> None:

        super(Resnet50, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(
            in_channels=num_channels, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.batch_norm1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU()
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(ResBlock, blocks=3, planes=64, stride=1)
        self.layer2 = self._make_layer(ResBlock, blocks=4, planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, blocks=6, planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, blocks=3, planes=512, stride=1, last=True)

        self.gem_pool = GeM()
        self.dropout = nn.Dropout(p=dropout)

        self.bn_fc = nn.BatchNorm1d(emb_dim)
        self.fc = nn.Linear(emb_dim, num_classes, bias=False)
        nn.init.kaiming_normal_(self.fc.weight)

    def _make_layer(self, ResBlock: Bottleneck, blocks: int, planes: int, stride: int = 1, last: bool = False):
        downsample = None
        if stride != 1 or self.in_channels != planes * ResBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * ResBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * ResBlock.expansion),
            )
        layers = []
        layers.append(
            ResBlock(in_channels=self.in_channels, out_channels=planes, stride=stride, downsample=downsample, last=last)
        )
        self.in_channels = planes * ResBlock.expansion
        for _ in range(1, blocks):
            layers.append(ResBlock(in_channels=self.in_channels, out_channels=planes, last=last))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        # Unsqueeze to simulate 1-channel image
        # print(f"inside resnet 50. x has shape {x.shape}")

        x = self.conv1(x.unsqueeze(1))
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.max_pool1(x)
        # print(f"inside resnet 50. before layer 1, x has shape {x.shape}")

        x = self.layer1(x)
        # print(f"inside resnet 50. after layer 1, x has shape {x.shape}")

        x = self.layer2(x)
        # print(f"inside resnet 50. after layer 2, x has shape {x.shape}")
        x = self.layer3(x)
        # print(f"inside resnet 50. after layer 3, x has shape {x.shape}")
        x = self.layer4(x)
        # print(f"inside resnet 50. after layer 4, x has shape {x.shape}")


        f_t = self.gem_pool(x)
        f_t = self.dropout(torch.flatten(f_t, start_dim=1))
        
        # print(f"inside resnet 50. after gempool, x has shape {f_t.shape}")


        f_c = self.bn_fc(f_t)
        cls = self.fc(f_c)

        return dict(f_t=f_t, f_c=f_c, cls=cls)
    



class TransformerEncoderModel(nn.Module):
    def __init__(
        self,
        emb_dim: int = 128,
        input_dim: int = 84,
        num_classes: int = 8858,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super(TransformerEncoderModel, self).__init__()
        
        # Linear layer to project input to desired embedding dimension
        self.input_proj = nn.Linear(input_dim, emb_dim)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Pooling layer
        self.gem_pool = GeM()
        self.dropout = nn.Dropout(p=dropout)

        # Fully connected classification layer
        self.bn_fc = nn.BatchNorm1d(emb_dim)
        self.fc = nn.Linear(emb_dim, num_classes, bias=False)
        nn.init.kaiming_normal_(self.fc.weight)

    def forward(self, x: torch.Tensor):
        # Project input to embedding dimensions
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, seq_len, input_dim)
        x = self.input_proj(x)  # Shape: (batch_size, seq_len, emb_dim)

        # Prepare input for Transformer
        x = x.permute(1, 0, 2)  # Transformer expects (seq_len, batch, emb_dim)

        # Transformer encoding
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # Global average pooling over sequence length

        # GeM Pooling
        f_t = self.gem_pool(x.unsqueeze(-1).unsqueeze(-1)).squeeze()  # Adapt to (batch_size, emb_dim)
        f_t = self.dropout(f_t)

        # Classification
        f_c = self.bn_fc(f_t)
        cls = self.fc(f_c)

        return dict(f_t=f_t, f_c=f_c, cls=cls)


class SimpleAttentionModel(nn.Module):
    def __init__(
        self,
        emb_dim: int = 156,
        num_classes: int = 8858,
        dropout=0.1,
        n_bins=84
    ) -> None:
        super(SimpleAttentionModel, self).__init__()

        # Linear projection layer
        self.projection = nn.Linear(n_bins, emb_dim, bias=False)
        
        # Attention scoring layer
        self.attention_weights = nn.Parameter(torch.randn(emb_dim))
        
        # Activation function
        self.activation = nn.ReLU()
        
        # BatchNorm and Dropout
        self.bn_fc = nn.BatchNorm1d(emb_dim)
        self.dropout = nn.Dropout(p=dropout)
        
        # Final fully connected layer for classification
        self.fc = nn.Linear(emb_dim, num_classes, bias=False)
        nn.init.kaiming_normal_(self.fc.weight)

    def forward(self, x: torch.Tensor):
        # Input x is of shape [batch_size, 84, 50]

        # Linear projection: [batch_size, 84, 50] -> [batch_size, 84, emb_dim]
        x = self.projection(x.transpose(1, 2))

        # Apply activation function
        x = self.activation(x)

        # Attention calculation: [batch_size, 84, emb_dim] -> [batch_size, 84]
        attention_scores = torch.matmul(x, self.attention_weights)
        attention_scores = F.softmax(attention_scores, dim=1)

        # Weighted sum of all vectors: [batch_size, emb_dim]
        x = torch.sum(x * attention_scores.unsqueeze(-1), dim=1)

        # Batch normalization and dropout
        x = self.bn_fc(x)
        x = self.dropout(x)

        # Classification layer
        cls = self.fc(x)

        return dict(f_t=x, f_c=x, cls=cls)
   