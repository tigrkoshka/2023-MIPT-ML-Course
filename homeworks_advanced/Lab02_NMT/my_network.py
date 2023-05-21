import numpy as np
import math
import torch
import torch.nn as nn

import random


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(
            num_embeddings=input_dim,
            embedding_dim=emb_dim
        )

        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, src):
        # src = [src sent len, batch size]

        # Compute an embedding from the src data and apply dropout to it
        # embedded = [src sent len, batch size, emb_dim]
        embedded = self.embedding(src)
        embedded = self.dropout(embedded)

        # Compute the RNN output values of the encoder RNN.
        # outputs, hidden and cell should be initialized here. Refer to nn.LSTM docs ;)
        #
        # outputs = [src sent len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]
        #
        # outputs are always from the top hidden layer
        output, (hidden, cell) = self.rnn(embedded)
        return output, hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(
            num_embeddings=output_dim,
            embedding_dim=emb_dim
        )

        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout
        )

        self.out = nn.Linear(
            in_features=hid_dim,
            out_features=output_dim
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input, hidden, cell):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hid dim]
        # context = [n layers, batch size, hid dim]
        # input = [1, batch size]
        input = input.unsqueeze(0)

        # Compute an embedding from the input data and apply dropout to it
        # embedded = [1, batch size, emb dim]
        embedded = self.dropout(self.embedding(input))

        # Compute the RNN output values of the encoder RNN. 
        # outputs, hidden and cell should be initialized here. Refer to nn.LSTM docs ;)

        # output = [sent len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # sent len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        # prediction = [batch size, output dim]
        prediction = self.out(output.squeeze(0))
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [src sent len, batch size]
        # trg = [trg sent len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        # Again, now batch is the first dimension instead of zero
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        _, hidden, cell = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        input = trg[0, :]

        for t in range(1, max_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input = (trg[t] if teacher_force else top1)

        return outputs


class Attention(nn.Module):
    def __init__(self, dimensions, attention_type='general'):
        super(Attention, self).__init__()

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)

        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, query, context):
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)

        if self.attention_type == "general":
            query = query.reshape(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.reshape(batch_size, output_len, dimensions)

        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())

        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)

        mix = torch.bmm(attention_weights, context)

        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)

        output = self.linear_out(combined).view(batch_size, output_len, dimensions)
        output = self.tanh(output)

        return output, attention_weights


class DecoderAttention(nn.Module):
    # This class is mainly copied from Decoder, all additions are marked with comments

    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout_p=0.5, attention_type="general"):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(
            num_embeddings=output_dim,
            embedding_dim=emb_dim
        )

        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout_p
        )

        # The following line is added
        self.attention = Attention(hid_dim, attention_type=attention_type)

        self.out = nn.Linear(
            in_features=hid_dim,
            out_features=output_dim
        )

        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, input, encoder_context, hidden, cell):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        # The following two lines are added (the attention itself)
        attention_output, _ = self.attention(output.transpose(0, 1), encoder_context.transpose(0, 1))
        attention_output = attention_output.transpose(0, 1)

        prediction = self.out(attention_output.squeeze(0))
        return prediction, hidden, cell


class Seq2SeqAttention(nn.Module):
    # This class is mainly copied from Seq2Seq, but checks
    # the DecoderAttention is used

    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert isinstance(self.decoder, DecoderAttention), \
            "Decoder must be an instance of DecoderAttention class!"

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size = tgt.shape[1]
        max_len = tgt.shape[0]
        tgt_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(max_len, batch_size, tgt_vocab_size).to(self.device)
        encoder_context, hidden, cell = self.encoder(src)
        input = tgt[0, :]

        for t in range(1, max_len):
            output, hidden, cell = self.decoder(input, encoder_context, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input = (tgt[t] if teacher_force else top1)

        return outputs


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, src):
        src = src + self.scale * self.pe[:src.size(0), :]
        return self.dropout(src)


class Seq2SeqTransformer(nn.Module):
    def __init__(self, src_dim, tgt_dim, emb_dim, enc_layers=1, dec_layers=1,
                 n_heads=1, dim_feedforward=2048, dropout=0.1, activation="relu",
                 pad_idx=1, sos_idx=2, device="cuda"):
        super(Seq2SeqTransformer, self).__init__()

        self.src_dim = src_dim
        self.tgt_dim = tgt_dim
        self.emb_dim = emb_dim

        self.enc_emb = nn.Embedding(src_dim, emb_dim)
        self.dec_emb = nn.Embedding(tgt_dim, emb_dim)
        self.pos_encoder = PositionalEncoding(emb_dim, dropout)

        self.transformer_model = nn.Transformer(d_model=emb_dim,
                                                nhead=n_heads,
                                                num_encoder_layers=enc_layers,
                                                num_decoder_layers=dec_layers,
                                                dim_feedforward=dim_feedforward,
                                                dropout=dropout,
                                                activation=activation)

        self.linear = nn.Linear(emb_dim, tgt_dim)

        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.device = device

        self.src_mask = None
        self.tgt_mask = None
        self.memory_mask = None

    def make_len_mask(self, inp):
        return inp == self.pad_idx

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        output_len = tgt.shape[0]
        batch_size = tgt.shape[1]
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)
        src_key_padding_mask = self.make_len_mask(src)

        src = (self.enc_emb(src) * math.sqrt(self.emb_dim)).transpose(0, 1)
        src = self.pos_encoder(src)

        if self.training:
            start_idxes = torch.ones([batch_size, 1], dtype=torch.int64, device=self.device) * self.sos_idx

            # adding the beginning symbol
            tgt = torch.cat((start_idxes, tgt), 1)[:, :-1]
            tgt = (self.dec_emb(tgt) * math.sqrt(self.emb_dim)).transpose(0, 1)
            tgt = self.pos_encoder(tgt)

            tgt_mask = self.transformer_model.generate_square_subsequent_mask(tgt.size(0)).to(self.device)

            decoder_outputs = self.transformer_model(
                src=src,
                tgt=tgt,
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_key_padding_mask)

            decoder_outputs = self.linear(decoder_outputs)

        else:
            encoder_hidden_states = self.transformer_model.encoder(src, src_key_padding_mask=src_key_padding_mask)

            decoder_inputs = torch.empty((src.size(1), output_len + 1),
                                         dtype=torch.int64, device=self.device).fill_(self.sos_idx)

            decoder_outputs = torch.zeros(output_len, src.size(1), self.tgt_dim, device=self.device)

            for i in range(output_len):
                decoder_input = (self.dec_emb(decoder_inputs[:, :i + 1]) * math.sqrt(self.emb_dim)).transpose(0, 1)
                decoder_input = self.pos_encoder(decoder_input)

                tgt_mask = self.transformer_model.generate_square_subsequent_mask(i + 1).to(self.device)

                decoder_output = self.transformer_model.decoder(
                    tgt=decoder_input,
                    memory=encoder_hidden_states,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=src_key_padding_mask)

                decoder_output = self.linear(decoder_output)[-1]
                decoder_outputs[i] = decoder_output
                decoder_inputs[:, i + 1] = decoder_output.max(1)[1]

        return decoder_outputs
