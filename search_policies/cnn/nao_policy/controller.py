import os
import logging

import IPython
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import Encoder, Encoder_Nasbench
from .decoder import Decoder, Decoder_Nasbench


SOS_ID = 0
EOS_ID = 0


class NAO(nn.Module):
    def __init__(self,
                 encoder_layers,
                 encoder_vocab_size,
                 encoder_hidden_size,
                 encoder_dropout,
                 encoder_length,
                 source_length,
                 encoder_emb_size,
                 mlp_layers,
                 mlp_hidden_size,
                 mlp_dropout,
                 decoder_layers,
                 decoder_vocab_size,
                 decoder_hidden_size,
                 decoder_dropout,
                 decoder_length,
                 ):
        super(NAO, self).__init__()
        self.encoder = Encoder(
            encoder_layers,
            encoder_vocab_size,
            encoder_hidden_size,
            encoder_dropout,
            encoder_length,
            source_length,
            encoder_emb_size,
            mlp_layers,
            mlp_hidden_size,
            mlp_dropout,
        )
        self.decoder = Decoder(
            decoder_layers,
            decoder_vocab_size,
            decoder_hidden_size,
            decoder_dropout,
            decoder_length,
            encoder_length
        )

        self.flatten_parameters()
    
    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()
    
    def forward(self, input_variable, target_variable=None):
        # IPython.embed(header='study nao sampler')
        # Input to encoder is so-called sequence.
        encoder_outputs, encoder_hidden, arch_emb, predict_value = self.encoder(input_variable)
        decoder_hidden = (arch_emb.unsqueeze(0), arch_emb.unsqueeze(0))
        decoder_outputs, decoder_hidden, ret = self.decoder(target_variable, decoder_hidden, encoder_outputs)
        # decoder_outputs 41 x [72,12], same as ret['sequence']
        decoder_outputs = torch.stack(decoder_outputs, 0).permute(1, 0, 2)
        # decoder_outputs becomes []
        arch = torch.stack(ret['sequence'], 0).permute(1, 0, 2)
        return predict_value, decoder_outputs, arch
    
    def generate_new_arch(self, input_variable, predict_lambda=1, direction='-'):
        encoder_outputs, encoder_hidden, arch_emb, predict_value, new_encoder_outputs, new_arch_emb = self.encoder.infer(
            input_variable, predict_lambda, direction=direction)
        new_encoder_hidden = (new_arch_emb.unsqueeze(0), new_arch_emb.unsqueeze(0))
        decoder_outputs, decoder_hidden, ret = self.decoder(None, new_encoder_hidden, new_encoder_outputs)
        new_arch = torch.stack(ret['sequence'], 0).permute(1, 0, 2)
        return new_arch


class NAO_Nasbench(NAO):
    def __init__(self,
                 encoder_layers,
                 encoder_vocab_size,
                 encoder_hidden_size,
                 encoder_dropout,
                 encoder_length,
                 source_length,
                 encoder_emb_size,
                 mlp_layers,
                 mlp_hidden_size,
                 mlp_dropout,
                 decoder_layers,
                 decoder_vocab_size,
                 decoder_hidden_size,
                 decoder_dropout,
                 decoder_length,
                 args=None
                 ):
        super(NAO, self).__init__()
        self.args = args
        self.encoder = Encoder_Nasbench(
            encoder_layers,
            encoder_vocab_size,
            encoder_hidden_size,
            encoder_dropout,
            encoder_length,
            source_length,
            encoder_emb_size,
            mlp_layers,
            mlp_hidden_size,
            mlp_dropout,
            args,
        )
        self.decoder = Decoder_Nasbench(
            decoder_layers,
            decoder_vocab_size,
            decoder_hidden_size,
            decoder_dropout,
            decoder_length,
            encoder_length,
            args
        )

        self.flatten_parameters()

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def forward(self, input_variable, target_variable=None):
        # IPython.embed(header='study nao_nasbench sampler, chaeck input and output')
        # Input to encoder is so-called sequence.
        encoder_outputs, encoder_hidden, arch_emb, predict_value = self.encoder(input_variable)
        decoder_hidden = (arch_emb.unsqueeze(0), arch_emb.unsqueeze(0))
        decoder_outputs, decoder_hidden, ret = self.decoder(target_variable, decoder_hidden, encoder_outputs)
        # decoder_outputs 41 x [72,12], same as ret['sequence']
        decoder_outputs = torch.stack(decoder_outputs, 0).permute(1, 0, 2)
        # decoder_outputs becomes []
        arch = torch.stack(ret['sequence'], 0).permute(1, 0, 2)
        return predict_value, decoder_outputs, arch

    def generate_new_arch(self, input_variable, predict_lambda=1, direction='-'):
        encoder_outputs, encoder_hidden, arch_emb, predict_value, new_encoder_outputs, new_arch_emb = self.encoder.infer(
            input_variable, predict_lambda, direction=direction)
        new_encoder_hidden = (new_arch_emb.unsqueeze(0), new_arch_emb.unsqueeze(0))
        decoder_outputs, decoder_hidden, ret = self.decoder(None, new_encoder_hidden, new_encoder_outputs)
        new_arch = torch.stack(ret['sequence'], 0).permute(1, 0, 2)
        return new_arch