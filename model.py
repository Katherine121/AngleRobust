import torch
from torch import nn
from einops import rearrange


def get_pad_mask(seq, pad_idx):
    """
    get the padding mask for indicating valid frames in the input sequence
    :param seq: shape of (b, len)
    :param pad_idx: 0
    :return: shape of (b, 1, len), if not equals to 0, set to 1, if equals to 0, set to 0
    """
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    """
    get the subsequent mask for masking the future frames
    :param seq: shape of (b, len)
    :return: lower triangle shape of (b, len, len)
    """
    b, s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, s, s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class FeatureEnhanceModule(nn.Module):
    def __init__(self, backbone):
        """
        Feature Enhance Module
        :param backbone: backbone of Feature Enhance Module (MobileNetV3 small)
        """
        super(FeatureEnhanceModule, self).__init__()
        self.backbone = backbone
        self.backbone.classifier = nn.Identity()

    def forward(self, x):
        """
        forward pass of Feature Enhance Module
        :param x: the provided input tensor
        :return: the visual semantic features of input
        """
        x = self.backbone(x)
        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        """
        pre normalization
        :param dim: input dimension of the last axis
        :param fn: next module
        """
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, mask, **kwargs):
        """
        forward pass of PreNorm
        :param x: the provided input tensor
        :return: the visual semantic features of input
        """
        return self.fn(self.norm(x), mask, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        """
        full connection layer.
        :param dim: input dimension.
        :param hidden_dim: hidden dimension.
        :param dropout: dropout rate.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask):
        """
        forward pass of FeedForward
        :param x: the provided input tensor
        :return: the visual semantic features of input
        """
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout=0.):
        """
        masked multi-head self attention.
        :param dim: input dimension.
        :param heads: the number of heads.
        :param dim_head: dimension of one head.
        :param dropout: dropout rate.
        """
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask):
        """
        forward pass of Attention
        :param x: the provided input tensor
        :param mask: padding and subsequent mask
        :return: the visual semantic features of input
        """
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if mask is not None:
            # for head axis broadcasting
            mask = mask.unsqueeze(1)
            dots = dots.masked_fill(mask == 0, -1e9)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class CrossAttentionModule(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        """
        Cross Attention Module.
        :param dim: input dimension.
        :param depth: depth of Cross Attention Module (Transformer Decoder).
        :param heads: the number of heads in Masked MSA.
        :param dim_head: dimension of one head.
        :param mlp_dim: hidden dimension in FeedForward.
        :param dropout: dropout rate.
        """
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, mask):
        """
        forward pass of Cross Attention Module
        :param x: the provided input tensor
        :param mask: padding and subsequent mask
        :return: the visual semantic features of input
        """
        for attn, ff in self.layers:
            x = attn(x, mask) + x
            x = ff(x, mask) + x
        return x


class ARTransformer(nn.Module):
    def __init__(self, *, backbone, extractor_dim,
                 num_classes1, num_classes2, len,
                 dim, depth, heads, dim_head, mlp_dim,
                 dropout=0., emb_dropout=0.):
        """
        ARTransformer
        :param backbone: backbone of Feature Enhance Module (MobileNetV3 small)
        :param extractor_dim: output dimension of Feature Enhance Module
        :param num_classes1: output dimension of ARTransformer
        :param num_classes2: output dimension of ARTransformer
        :param len: input sequence length of ARTransformer
        :param dim: input dimension of Cross Attention Module
        :param depth: depth of Cross Attention Module
        :param heads: the number of heads in Multi-Head Self Attention layer
        :param dim_head: dimension of one head
        :param mlp_dim: hidden dimension in FeedForward layer
        :param dropout: dropout rate
        :param emb_dropout: dropout rate after position embedding
        """
        super().__init__()
        self.extractor = FeatureEnhanceModule(backbone)
        self.extractor_dim = extractor_dim
        self.dim = dim
        self.len = len

        self.ang_linear = nn.Linear(num_classes2, dim)
        # extractor_dim+dim
        self.img_linear = nn.Linear(self.extractor_dim + dim, dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, self.len, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = CrossAttentionModule(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.mlp_head = nn.Linear(dim, 2 * dim)
        self.head_label = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes1)
        )
        self.head_target = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes1)
        )
        self.head_angle = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Hardtanh(),
            nn.Linear(dim, dim // 2),
            nn.Hardtanh(),
            nn.Linear(dim // 2, num_classes2)
        )

    def forward(self, img, ang):
        """
        forward pass of ARTransformer
        :param img: input frame sequence
        :param ang: input angle sequence
        :return: the current position preds, the next position preds, the direction angle preds
        """
        # b,1,len
        src_mask1 = get_pad_mask(img[:, :, 0, 0, 0].view(-1, self.len), pad_idx=0)
        # b,len,len
        b = img.size(0)
        src_mask = src_mask1 & get_subsequent_mask(img[:, :, 0, 0, 0].view(-1, self.len))

        # b,len,3,224,224->b*len,3,224,224->b*len,576->b,len,576
        img = self.extractor(img.view(-1, 3, 224, 224))
        img = img.view(-1, self.len, self.extractor_dim)

        # b,len,2->b,len,dim
        for i in range(1, self.len):
            ang[:, i, :] += ang[:, i - 1, :]
        ang = self.ang_linear(ang)

        # b,len,extractor_dim+dim->b,len,dim
        img = torch.cat((img, ang), dim=-1)
        img = self.img_linear(img)

        img += self.pos_embedding
        img = self.dropout(img)

        img = self.transformer(img, src_mask)

        # b,len,dim->b,dim
        res = torch.ones(b, self.dim).to(img.device)
        for i in range(0, b):
            # len,dim
            pic = img[i]
            # dim
            res[i] = pic[src_mask1[i].sum() - 1]

        # b,dim->b,2*dim->2,b,dim
        img = self.mlp_head(res)
        ang = img[:, self.dim:]
        img = img[:, 0: self.dim]
        return self.head_label(img), self.head_target(img), self.head_angle(ang)
