import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer.decoder import Decoder
from transformer.multihead_attention import MultiHeadAttention
from transformer.positional_encoding import PositionalEncoding
from transformer.pointerwise_feedforward import PointerwiseFeedforward
from transformer.encoder_decoder import EncoderDecoder
from transformer.encoder import Encoder
from transformer.encoder_layer import EncoderLayer
from transformer.decoder_layer import DecoderLayer
from transformer.batch import subsequent_mask
import numpy as np
import scipy.io
import os

import copy
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

        c = copy.deepcopy
        attn = MultiHeadAttention(h, d_model)
        ff = PointerwiseFeedforward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        self.generator = nn.ModuleDict({
            'src_embed': nn.Sequential(LinearEmbedding(enc_inp_size + z_dim, d_model), c(position)),
            'encoder': Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            'tgt_embed': nn.Sequential(nn.Linear(d_model, dec_out_size)),
        })

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in self.generator.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def sample_noise(self, batch_size):
        noise = torch.randn(batch_size, self.src_len, self.z_dim, device=self.device)
        # noise[:, 0, -1] = 1.  # Distinguish start-of-sequence token
        return noise

    def forward(self, src, noise):
        """
        Given a src trajectory in shape ((b)atch, self.src_len, (d)iminsionality)
        Generate a tgt trajectory in shape ((b)atch, self.tgt_len, (d)iminsionality)
        """
        batch_size = src.shape[0]
        src_mask = torch.ones((batch_size, 1, self.src_len)).to(self.device)
        enc_inp = torch.cat((src, noise), dim=-1)
        return self.generator['tgt_embed'](
                self.generator['encoder'](
                    self.generator['src_embed'](enc_inp), src_mask))[:, -self.tgt_len:, :]

class Critic(nn.Module):
    def __init__(self, disc_inp_size, disc_seq_len, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, device='cpu'):
        super(Critic, self).__init__()

        self.device = device

        c = copy.deepcopy
        attn = MultiHeadAttention(h, d_model)
        ff = PointerwiseFeedforward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        self.critic = nn.ModuleDict({
            'src_embed': nn.Sequential(LinearEmbedding(disc_inp_size, d_model), c(position)),
            'encoder': Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            'disc_head': nn.Sequential(nn.Flatten(), nn.Linear(d_model, 1)),
        })

        for p in self.critic.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, seq):
        """
        Returns the probability that this sequence is Real
        """
        mask = torch.ones((seq.shape[0], 1, seq.shape[1])).to(self.device)
        return self.critic['disc_head'](
                self.critic['encoder'](
                    self.critic['src_embed'](seq), mask)[:, -1, :])

class LinearEmbedding(nn.Module):
    def __init__(self, inp_size,d_model):
        super(LinearEmbedding, self).__init__()
        # lut => lookup table
        self.lut = nn.Linear(inp_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class TFHeadGenerator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, out_size):
        super(TFHeadGenerator, self).__init__()
        self.proj = nn.Linear(d_model, out_size)

    def forward(self, x):
        return self.proj(x)


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
    gen_loss += lambda_recon * torch.norm(generations - reals)
    return gen_loss

def get_crit_loss(crit, src, real, fake, crit_fake_pred, crit_real_pred, c_lambda):
    """
    Returns the W-Loss
    """
    batch_size = src.shape[0]
    epsilon = torch.rand(batch_size, 1, 1, device=crit.device, requires_grad=True)
    gradient = get_gradient(crit, src, real, fake, epsilon)
    gp = gradient_penalty(gradient)
    crit_loss = -crit_real_pred.mean() + crit_fake_pred.mean() + gp * c_lambda
    return crit_loss
