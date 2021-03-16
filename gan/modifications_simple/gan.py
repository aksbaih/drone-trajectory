import torch
import torch.nn as nn
import math


class Generator(nn.Module):
    def __init__(self, src_len, tgt_len, enc_inp_size, dec_inp_size, dec_out_size, z_dim, N=6,
                   d_model=512, d_ff=2048, h=8, dropout=0.1, device='cpu'):
        super(Generator, self).__init__()
        self.device = device
        self.src_len = src_len
        self.tgt_len = tgt_len
        self.dec_inp_size = dec_inp_size
        self.z_dim = z_dim

        self.gen = nn.Sequential(
            nn.Linear(enc_inp_size + z_dim, d_model),
            PositionalEncoding(d_model=d_model, dropout=dropout),
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=h, dim_feedforward=d_ff,
                                                             dropout=dropout, activation='gelu'),
                                  num_layers=N),
            nn.Linear(d_model, dec_out_size),
        )

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in self.gen.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def sample_noise(self, batch_size):
        noise = torch.randn(batch_size, self.src_len, self.z_dim, device=self.device)
        return noise

    def forward(self, src, noise):
        """
        Given a src trajectory in shape ((b)atch, self.src_len, (d)iminsionality)
        Generate a tgt trajectory in shape ((b)atch, self.tgt_len, (d)iminsionality)
        """
        enc_inp = torch.cat((src, noise), dim=-1)
        return self.gen(enc_inp)[:, -self.tgt_len:, :]

class Critic(nn.Module):
    def __init__(self, disc_inp_size, disc_seq_len, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, device='cpu'):
        super(Critic, self).__init__()

        self.device = device

        self.crit = nn.Sequential(
            nn.Linear(disc_inp_size, d_model),
            PositionalEncoding(d_model=d_model, dropout=dropout),
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=h, dim_feedforward=d_ff,
                                                             dropout=dropout, activation='gelu'),
                                  num_layers=N, norm=nn.LayerNorm([disc_seq_len, d_model])),
            nn.Flatten(),
            nn.Linear(d_model * disc_seq_len, 1)
        )

        for p in self.crit.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, seq):
        return self.crit(seq)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def gradient_penalty(gradient):
    gradient = gradient.view(len(gradient), -1)
    gradient_norm = gradient.norm(2, dim=1)
    penalty = (gradient_norm - 1.).pow(2).mean()
    return penalty

def get_gradient(crit, src, real, fake, epsilon):
    mixed_seqs = real * epsilon + fake * (1 - epsilon)
    seq = torch.cat((src, mixed_seqs), dim=1)
    mixed_scores = crit(seq)
    gradient = torch.autograd.grad(
        inputs=seq,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient

def get_gen_loss(crit_fake_pred, generations, reals, lambda_recon):
    gen_loss = -crit_fake_pred.mean()
    # gen_loss += generations[..., -1].pow(2).sum()  # we want it to be 0 for any state other than the noise
    gen_loss += lambda_recon * torch.pow(generations - reals, 2).mean()
    return gen_loss

def get_crit_loss(crit, src, real, fake, crit_fake_pred, crit_real_pred, c_lambda, lambda_recon):
    """
    Returns the W-Loss
    """
    batch_size = src.shape[0]
    epsilon = torch.rand(batch_size, 1, 1, device=crit.device, requires_grad=True)
    gradient = get_gradient(crit, src, real, fake, epsilon)
    gp = gradient_penalty(gradient)
    crit_loss = -crit_real_pred.mean() + crit_fake_pred.mean() + gp * c_lambda
    return crit_loss

#
# CUDA_VISIBLE_DEVICES=0 python train_gan.py     --dataset_folder ../../dataset     --dataset_name data     --name simple_long --stop_recon 24 --z_dim 3     --obs 12 --preds 8     --val_size 64     --max_epoch 36000     --save_step 1     --visual_step 30     --grad_penality 10  --lambda_recon 0.01   --crit_repeats 4    --batch_size 128
