
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from param import args
from DAN_modules.refer_find_modules import REFER, FIND
from DAN_modules.language_model import WordEmbedding, DynamicRNN

class EncoderLSTM(nn.Module):
    ''' Encodes navigation instructions, returning hidden state context (for
        attention methods) and a decoder initial state. '''

    def __init__(self, vocab_size, embedding_size, hidden_size, padding_idx, 
                            dropout_ratio, bidirectional=False, num_layers=1):
        super(EncoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(p=dropout_ratio)
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers

        self.word_embed = nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.sent_embed = nn.LSTM(embedding_size, hidden_size, 2, dropout=dropout_ratio, batch_first=True)
        self.sent_embed = DynamicRNN(self.sent_embed)
        self.hist_embed = nn.LSTM(embedding_size, hidden_size, 2, dropout=dropout_ratio, batch_first=True)
        self.hist_embed = DynamicRNN(self.hist_embed)

        self.encoder2decoder = nn.Linear(2 * hidden_size * self.num_directions,
            hidden_size * self.num_directions)

        self.layer_stack = nn.ModuleList([
            REFER(d_model=512, d_inner=1024, n_head=4, d_k=256, d_v=256, dropout=0.2)
            for _ in range(2)])

    def init_state(self, inputs):
        ''' Initialize to zero cell states and hidden states.'''
        batch_size = inputs.size(0)
        h0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        c0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        return h0.cuda(), c0.cuda()

    def add_entry(self, mem, hist, hl):
        h_emb = self.word_embed(hist)
        h_emb = self.drop(h_emb)
        _, h_emb, _ = self.hist_embed(h_emb, hl)
        h_emb = h_emb.unsqueeze(1)

        if mem is None: mem = h_emb
        else: mem = torch.cat((mem, h_emb), 1)
        return mem

    def refer_module(self, mem, q):
        '''
        q : [b, 512]
        mem : [b, number of entry, 512]
        '''
        context = q.unsqueeze(1)
        for enc_layer in self.layer_stack:
            context, _ = enc_layer(context, mem)
        return context.squeeze(1)

    def forward(self, inputs, lengths, Last_QA, Last_QA_lengths, hist, hist_lengths, tar, tar_lengths):
        ''' Expects input vocab indices as (batch, seq_len). Also requires a 
            list of lengths for dynamic batching. '''

        q = Last_QA
        ql = Last_QA_lengths
        c = tar
        cl = tar_lengths
        h = hist
        hl = hist_lengths
        # write history embedding to memory
        mem = self.add_entry(None, c, cl)
        enc_outs = []
        q_emb = self.word_embed(q)
        q_emb = self.drop(q_emb)

        ctx, q_emb, c_t = self.sent_embed(q_emb, ql)

        for i in range(15):
            
            his = self.refer_module(mem, q_emb)
            ref_aware = torch.cat((q_emb, his), 1)
            enc_outs.append(ref_aware)

            # write history embedding to memory
            if i != 14:
                mem = self.add_entry(mem, h[:, i, :], hl[:, i])

        enc_out = torch.stack(enc_outs, 1)
        # enc_out = self.linear(enc_out[:, -1, :])
        decoder_init = nn.Tanh()(self.encoder2decoder(enc_out[:, -1, :]))
        
        mem = torch.cat((mem, q_emb.unsqueeze(1)), 1)
        mem = self.drop(mem)

        return mem, decoder_init, c_t


class SoftDotAttention(nn.Module):
    '''Soft Dot Attention.

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    '''

    def __init__(self, query_dim, ctx_dim):
        '''Initialize layer.'''
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(query_dim, ctx_dim, bias=False)
        self.sm = nn.Softmax()
        self.linear_out = nn.Linear(query_dim + ctx_dim, query_dim, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, h, context, mask=None,
                output_tilde=True, output_prob=True):
        '''Propagate h through the network.

        h: batch x dim
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        logit = attn

        if mask is not None:
            # -Inf masking prior to the softmax
            attn.masked_fill_(mask, -float('inf'))
        # attn = self.sm(attn, dim=1)    # There will be a bug here, but it's actually a problem in torch source code.
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        if not output_prob:
            attn = logit
        if output_tilde:
            h_tilde = torch.cat((weighted_context, h), 1)
            h_tilde = self.tanh(self.linear_out(h_tilde))
            return h_tilde, attn
        else:
            return weighted_context, attn


class AttnDecoderLSTM(nn.Module):
    ''' An unrolled LSTM with attention over instructions for decoding navigation actions. '''

    def __init__(self, embedding_size, hidden_size,
                       dropout_ratio, feature_size=2048+4):
        super(AttnDecoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.embedding = nn.Sequential(
            nn.Linear(args.angle_feat_size, self.embedding_size),
            nn.Tanh()
        )
        self.drop = nn.Dropout(p=dropout_ratio)
        self.drop_env = nn.Dropout(p=args.featdropout)
        self.lstm = nn.LSTMCell(embedding_size+feature_size, hidden_size)
        self.feat_att_layer = SoftDotAttention(hidden_size, feature_size)
        self.attention_layer = SoftDotAttention(hidden_size, hidden_size)
        self.candidate_att_layer = SoftDotAttention(hidden_size, feature_size)

    def forward(self, action, feature, cand_feat,
                h_0, prev_h1, c_0,
                ctx, ctx_mask=None,
                already_dropfeat=False):
        '''
        Takes a single step in the decoder LSTM (allowing sampling).
        action: batch x angle_feat_size
        feature: batch x 36 x (feature_size + angle_feat_size)
        cand_feat: batch x cand x (feature_size + angle_feat_size)
        h_0: batch x hidden_size
        prev_h1: batch x hidden_size
        c_0: batch x hidden_size
        ctx: batch x seq_len x dim
        ctx_mask: batch x seq_len - indices to be masked
        already_dropfeat: used in EnvDrop
        '''
        action_embeds = self.embedding(action)

        # Adding Dropout
        action_embeds = self.drop(action_embeds)

        if not already_dropfeat:
            # Dropout the raw feature as a common regularization
            feature[..., :-args.angle_feat_size] = self.drop_env(feature[..., :-args.angle_feat_size])   # Do not drop the last args.angle_feat_size (position feat)

        prev_h1_drop = self.drop(prev_h1)
        attn_feat, _ = self.feat_att_layer(prev_h1_drop, feature, output_tilde=False)
        
        concat_input = torch.cat((action_embeds, attn_feat), 1)
        
        h_1, c_1 = self.lstm(concat_input, (prev_h1, c_0))

        h_1_drop = self.drop(h_1)
        h_tilde, alpha = self.attention_layer(h_1_drop, ctx, ctx_mask)

        # Adding Dropout
        h_tilde_drop = self.drop(h_tilde)

        if not already_dropfeat:
            cand_feat[..., :-args.angle_feat_size] = self.drop_env(cand_feat[..., :-args.angle_feat_size])

        _, logit = self.candidate_att_layer(h_tilde_drop, cand_feat, output_prob=False)

        return h_1, c_1, logit, h_tilde