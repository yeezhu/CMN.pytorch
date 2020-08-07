import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class DynamicRNN(nn.Module):
    """
    This code is modified from batra-mlp-lab's repository.
    https://github.com/batra-mlp-lab/visdial-challenge-starter-pytorch
    """
    def __init__(self, rnn_model):
        super(DynamicRNN, self).__init__()
        self.rnn_model = rnn_model

    def forward(self, seq_input, seq_lens, initial_state=None):
        """A wrapper over pytorch's rnn to handle sequences of variable length.

        Arguments
        ---------
        seq_input : torch.autograd.Variable
            Input sequence tensor (padded) for RNN model. (b, max_seq_len, embed_size)
        seq_lens : torch.LongTensor
            Length of sequences (b, )
        initial_state : torch.autograd.Variable
            Initial (hidden, cell) states of RNN model.

        Returns
        -------
            A single tensor of shape (batch_size, rnn_hidden_size) corresponding
            to the outputs of the RNN model at the last time step of each input
            sequence.
        """
        sorted_len, fwd_order, bwd_order = self._get_sorted_order(seq_lens)
        sorted_seq_input = seq_input.index_select(0, fwd_order)
        packed_seq_input = pack_padded_sequence(
            sorted_seq_input, lengths=sorted_len, batch_first=True)

        if initial_state is not None:
            hx = initial_state
            sorted_hx = [x.index_select(1, fwd_order) for x in hx]
            assert hx[0].size(0) == self.rnn_model.num_layers
        else:
            hx = None
        self.rnn_model.flatten_parameters()
        enc_h, (h_n, c_n) = self.rnn_model(packed_seq_input, hx)
        ctx, _ = pad_packed_sequence(enc_h, batch_first=True)

        c_t = c_n[-1].index_select(dim=0, index=bwd_order)
        rnn_output = h_n[-1].index_select(dim=0, index=bwd_order)
        return ctx, rnn_output, c_t

    @staticmethod
    def _get_sorted_order(lens):
        sorted_len, fwd_order = torch.sort(lens.contiguous().view(-1), 0, descending=True)
        _, bwd_order = torch.sort(fwd_order)
        if isinstance(sorted_len, Variable):
            sorted_len = sorted_len.data
        sorted_len = list(sorted_len)
        return sorted_len, fwd_order, bwd_order


class WordEmbedding(nn.Module):
    """
    code from @jnhwkim (Jin-Hwa Kim)
    https://github.com/jnhwkim/ban-vqa
    """
    def __init__(self, ntoken, emb_dim, dropout, padding_idx):
        super(WordEmbedding, self).__init__()
        self.emb = nn.Embedding(ntoken+1, emb_dim, padding_idx)
        self.dropout = nn.Dropout(dropout)
        self.ntoken = ntoken
        self.emb_dim = emb_dim

    def init_embedding(self, np_file, tfidf=None, tfidf_weights=None):
        weight_init = torch.from_numpy(np.load(np_file))
        assert weight_init.shape == (self.ntoken, self.emb_dim)
        self.emb.weight.data[:self.ntoken] = weight_init

    def forward(self, x):
        emb = self.emb(x)
        emb = self.dropout(emb)
        return emb
