import torch
import torchvision.models.vision_transformer
from torch import nn
from einops import rearrange


def get_pad_mask(seq, pad_idx):
    """
    get the padding mask for indicating valid frames in the input sequence.
    :param seq: shape of (b, len).
    :param pad_idx: 0.
    :return: shape of (b, 1, len), if not equals to 0, set to 1, if equals to 0, set to 0.
    """
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    """
    get the subsequent mask for masking the future frames.
    :param seq: shape of (b, len).
    :return: lower triangle shape of (b, len, len).
    """
    b, s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, s, s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class SequenceFeatureEncoder(nn.Module):
    def __init__(self, backbone):
        """
        Sequence Feature Encoder.
        :param backbone: backbone of Sequence Feature Encoder (MobileNetV3 small).
        """
        super(SequenceFeatureEncoder, self).__init__()
        self.backbone = backbone
        self.backbone.classifier = nn.Identity()

    def forward(self, x):
        """
        forward pass of Sequence Feature Encoder.
        :param x: the provided input tensor.
        :return: the visual semantic features of an image.
        """
        x = self.backbone(x)
        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        """
        pre normalization.
        :param dim: input dimension of the last axis.
        :param fn: next module.
        """
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, mask, **kwargs):
        """
        forward pass of PreNorm.
        :param x: the provided input tensor.
        :return: the visual semantic features of input.
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
        forward pass of FeedForward.
        :param x: the provided input tensor.
        :return: the visual semantic features of input.
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
        forward pass of Attention.
        :param x: the provided input tensor.
        :param mask: padding and subsequent mask.
        :return: the visual semantic features of input.
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


class CrossAttentionMixer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        """
        Cross Attention Mixer.
        :param dim: input dimension.
        :param depth: depth of Cross Attention Mixer (Transformer Decoder).
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
        forward pass of Cross Attention Mixer.
        :param x: the provided input tensor.
        :param mask: padding and subsequent mask.
        :return: the visual semantic features of input.
        """
        for attn, ff in self.layers:
            x = attn(x, mask) + x
            x = ff(x, mask) + x
        return x


class AREncoder(nn.Module):
    def __init__(self, backbone, extractor_dim, len,
                 dim, heads, mlp_dim, depth):
        super().__init__()
        self.extractor = SequenceFeatureEncoder(backbone)
        self.extractor_dim = extractor_dim
        self.len = len

        # extractor_dim+dim
        self.img_linear = nn.Linear(self.extractor_dim + 2, dim)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads,
                                                        dim_feedforward=mlp_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=depth)

    def forward(self, img, ang):
        # b,len,3,224,224->b*len,3,224,224->b*len,576->b,len,576
        img = self.extractor(img.view(-1, 3, 224, 224))
        img = img.view(-1, self.len, self.extractor_dim)

        # b,len,extractor_dim+2->b,len,dim
        img = torch.cat((img, ang), dim=-1)
        img = self.img_linear(img)

        img = self.transformer(img)
        return img


class ARTransformer(nn.Module):
    def __init__(self, *, backbone, extractor_dim,
                 num_classes1, num_classes2, len,
                 dim, heads, mlp_dim, depth,
                 moco_m, moco_tau, moco_criterion):
        """
        ARTransformer
        :param backbone: backbone of Sequence Feature Encoder (MobileNetV3 small).
        :param extractor_dim: output dimension of Sequence Feature Encoder.
        :param num_classes1: output dimension of ARTransformer.
        :param num_classes2: output dimension of ARTransformer.
        :param len: input sequence length of ARTransformer.
        :param dim: input dimension of Cross Attention Mixer.
        :param depth: depth of Cross Attention Mixer.
        :param heads: the number of heads in Masked MSA.
        :param dim_head: dimension of one head.
        :param mlp_dim: hidden dimension in FeedForward.
        :param dropout: dropout rate.
        :param emb_dropout: dropout rate after position embedding.
        """
        super().__init__()
        self.encoder_q = AREncoder(backbone, extractor_dim, len, dim, heads, mlp_dim, depth)
        self.encoder_k = AREncoder(backbone, extractor_dim, len, dim, heads, mlp_dim, depth)

        self.extractor_dim = extractor_dim
        self.dim = dim
        self.len = len
        self.m = moco_m
        self.tau = moco_tau
        self.criterion = moco_criterion

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

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def header(self, img):
        img = img[:, -1, :]
        return self.head_label(img), self.head_target(img), self.head_angle(img)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def forward(self, img, cont_img, ang):
        """
        forward pass of ARTransformer.
        :param img: input frame sequence.
        :param ang: input angle sequence.
        :return: the current position preds, the next position preds, the direction angle preds.
        """
        # train
        if cont_img is not None:
            img_q1 = self.encoder_q(img, ang)
            img_q2 = self.encoder_q(cont_img, ang)

            q1 = nn.functional.normalize(img_q1, dim=-1)
            q1 = q1.view(-1, self.dim)
            q2 = nn.functional.normalize(img_q2, dim=-1)
            q2 = q2.view(-1, self.dim)

            # compute key features
            with torch.no_grad():  # no gradient to keys
                self._momentum_update_key_encoder()  # update the key encoder

                img_k1 = self.encoder_k(img, ang)
                img_k2 = self.encoder_k(cont_img, ang)  # keys: NxC

                k1 = nn.functional.normalize(img_k1, dim=-1)
                k1 = k1.view(-1, self.dim)
                k2 = nn.functional.normalize(img_k2, dim=-1)
                k2 = k2.view(-1, self.dim)

            qk_loss = 0.5 * ctr(q1, k2, self.criterion, self.tau) + \
                      0.5 * ctr(q2, k1, self.criterion, self.tau)

            img_label, img_target, img_angle = self.header(img_q1)
            cont_img_label, cont_img_target, cont_img_angle = self.header(img_q2)

            return img_label, img_target, img_angle, \
                   cont_img_label, cont_img_target, cont_img_angle, qk_loss

        # test
        else:
            img_q1 = self.encoder_q(img, ang)
            img_label, img_target, img_angle = self.header(img_q1)
            return img_label, img_target, img_angle


def ctr(q, k, criterion, tau):
    logits = torch.mm(q, k.t())
    N = q.size(0)
    labels = range(N)
    labels = torch.LongTensor(labels).cuda()
    loss = criterion(logits / tau, labels)
    return 2 * tau * loss
