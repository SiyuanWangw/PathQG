import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('..')
sys.path.append('../..')
from Attention import BahdanauAttention


class Selector(nn.Module):
    def __init__(self, args, pad_idx=0, pretrain_src_embeddings=None, pretrain_tgt_embeddings=None):
        super(Selector, self).__init__()
        self.args = args
        self.init_embedding(pad_idx, pretrain_src_embeddings, pretrain_tgt_embeddings)
        self.src_encoding_lstm = nn.LSTM(self.args.embedding_size + self.args.tagging_embedding_size * 2,
                                         self.args.encoder_hidden_size,
                                         num_layers=self.args.num_layers,
                                         bidirectional=True,
                                         batch_first=True)

        self.encoder_forward_lstm_cell = nn.LSTMCell(self.args.embedding_size,
                                                     self.args.entity_encoder_hidden_size)
        self.encoder_backward_lstm_cell = nn.LSTMCell(self.args.embedding_size,
                                                      self.args.entity_encoder_hidden_size)

        self.dropout = nn.Dropout(self.args.dropout)

        self.sentence_attn = BahdanauAttention(self.args.entity_encoder_hidden_size * 2,
                                               self.args.encoder_hidden_size * 2,
                                               self.args.attn_size)
        self.entity_tgt_dense = nn.Linear(self.args.entity_encoder_hidden_size * 2 + self.args.encoder_hidden_size * 2,
            self.args.entity_dense_size)
        self.entity_out = nn.Linear(self.args.entity_dense_size, self.args.entity_tag_num)
        torch.nn.init.xavier_uniform_(self.entity_tgt_dense.weight)
        torch.nn.init.xavier_uniform_(self.entity_out.weight)

    def init_embedding(self, pad_idx, pretrain_src_embeddings, pretrain_tgt_embeddings):
        self.src_embedding = nn.Embedding(self.args.src_vocab_size, self.args.embedding_size, padding_idx=pad_idx)
        if pretrain_src_embeddings is not None:
            self.src_embedding.weight = nn.Parameter(torch.FloatTensor(pretrain_src_embeddings))
        else:
            self.src_embedding.weight.data.uniform_(-0.1, 0.1)
        self.tgt_embedding = nn.Embedding(self.args.tgt_vocab_size, self.args.embedding_size, padding_idx=pad_idx)
        if pretrain_tgt_embeddings is not None:
            self.tgt_embedding.weight = nn.Parameter(torch.FloatTensor(pretrain_tgt_embeddings))
        else:
            self.tgt_embedding.weight.data.uniform_(-0.1, 0.1)

        self.answer_tagging_embeddings = nn.Embedding(self.args.answer_class_num, self.args.tagging_embedding_size)
        self.answer_tagging_embeddings.weight.data.uniform_(-0.1, 0.1)
        self.end_tagging_embeddings = nn.Embedding(self.args.end_tag_num, self.args.tagging_embedding_size)
        self.end_tagging_embeddings.weight.data.uniform_(-0.1, 0.1)

    def src_encoding(self, src_input, src_input_lengths, answer_tag_input, end_tag_input):
        # encoding of src input
        sorted_src_input_length, idx_sort = torch.sort(src_input_lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0, descending=False)
        src_input = src_input[idx_sort]
        answer_tag_input = answer_tag_input[idx_sort]
        end_tag_input = end_tag_input[idx_sort]

        src_input_embeddings = self.dropout(self.src_embedding(src_input))
        answer_tag_input_embeddings = self.dropout(self.answer_tagging_embeddings(answer_tag_input))
        end_tag_input_embeddings = self.dropout(self.end_tagging_embeddings(end_tag_input))

        input_embeddings = torch.cat([src_input_embeddings, answer_tag_input_embeddings, end_tag_input_embeddings], dim=-1)
        packed_input_embeddings = nn.utils.rnn.pack_padded_sequence(input_embeddings, sorted_src_input_length, batch_first=True)
        encoder_outputs, (h_t, c_t) = self.src_encoding_lstm(packed_input_embeddings)
        encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(encoder_outputs, batch_first=True)

        encoder_outputs = encoder_outputs[idx_unsort]
        h_t = torch.cat([h_t[0][idx_unsort].unsqueeze(0), h_t[1][idx_unsort].unsqueeze(0)], 0)
        c_t = torch.cat([c_t[0][idx_unsort].unsqueeze(0), c_t[1][idx_unsort].unsqueeze(0)], 0)

        init_h = torch.cat((h_t[0], h_t[1]), dim=-1)
        init_c = torch.cat((c_t[0], c_t[1]), dim=-1)
        init_state = (init_h, init_c)

        return encoder_outputs, init_state

    def select(self, encoder_outputs, average_entity_embedding):

        fw_outputs = []
        bw_outputs = []
        for t in range(average_entity_embedding.size(1)):
            fw_cur_input = average_entity_embedding[:, t]
            fw_cur_input = self.dropout(fw_cur_input)
            bw_cur_input = average_entity_embedding[:, -(t + 1)]
            bw_cur_input = self.dropout(bw_cur_input)

            if t == 0:
                fw_h_t, fw_c_t = self.encoder_forward_lstm_cell(fw_cur_input)
                bw_h_t, bw_c_t = self.encoder_backward_lstm_cell(bw_cur_input)
            else:
                fw_h_t, fw_c_t = self.encoder_forward_lstm_cell(fw_cur_input, fw_init_hidden)
                bw_h_t, bw_c_t = self.encoder_backward_lstm_cell(bw_cur_input, bw_init_hidden)

            fw_init_hidden = (fw_h_t, fw_c_t)
            bw_init_hidden = (bw_h_t, bw_c_t)

            fw_outputs.append(fw_h_t.unsqueeze(1))
            bw_outputs.append(bw_h_t.unsqueeze(1))

        bw_outputs.reverse()
        fw_outputs = torch.cat(fw_outputs, 1)
        bw_outputs = torch.cat(bw_outputs, 1)
        outputs = torch.cat([fw_outputs, bw_outputs], -1)

        contexts = []
        for t in range(outputs.size(1)):
            _, cur_context = self.sentence_attn(outputs[:, t], encoder_outputs)
            contexts.append(cur_context)
        contexts = torch.cat(contexts, 1)

        outputs = torch.cat([outputs, contexts], -1)

        dense_outputs = F.relu(self.entity_tgt_dense(outputs))
        logits = self.entity_out(dense_outputs)

        return logits

    def forward(self, src_input, src_input_lengths, answer_tag_input, end_tag_input, entity_seq_input, entity_type_seq_input, tgt_input, tgt_input_lengths, answer_input):
        encoder_outputs, init_state = self.src_encoding(src_input, src_input_lengths, answer_tag_input, end_tag_input)

        mask = torch.sign(entity_seq_input).float().unsqueeze(-1)
        entity_embedding = self.src_embedding(entity_seq_input)
        average_entity_embedding = torch.sum(mask.mul(entity_embedding), 2) / torch.sum(mask, 2).clamp(1)
        lstm_entity_embedding = average_entity_embedding

        logits = self.select(encoder_outputs, lstm_entity_embedding)
        return logits

