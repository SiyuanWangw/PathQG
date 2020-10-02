import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
sys.path.append('..')
sys.path.append('../..')
from Attention import BahdanauAttention
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


class Generator(nn.Module):
    def __init__(self, args, pad_idx=0, pretrain_src_embeddings=None, pretrain_tgt_embeddings=None):
        super(Generator, self).__init__()
        self.args = args
        self.bs = BeamSearch(self.args)
        self.init_embedding(pad_idx, pretrain_src_embeddings, pretrain_tgt_embeddings)

        self.encoder_lstm = nn.LSTM(self.args.embedding_size + self.args.tagging_embedding_size*2,
                                    self.args.encoder_hidden_size,
                                    num_layers=self.args.num_layers,
                                    bidirectional=True,
                                    dropout=self.args.dropout,
                                    batch_first=True)
        print(self.encoder_lstm)
        if not self.args.attn_input_feed:
            self.decoder_lstm_cell = nn.LSTMCell(self.args.embedding_size,
                                                 self.args.decoder_hidden_size)
        else:
            self.decoder_lstm_cell = nn.LSTMCell(self.args.embedding_size + self.args.decoder_hidden_size,
                                                 self.args.decoder_hidden_size)
        self.sentence_attn = BahdanauAttention(self.args.decoder_hidden_size, self.args.encoder_hidden_size*2, self.args.attn_size)
        self.dropout = nn.Dropout(self.args.dropout)

        self.dense = nn.Linear(self.args.decoder_hidden_size+self.args.encoder_hidden_size*2, self.args.question_dense_size)
        self.out = nn.Linear(self.args.question_dense_size, self.args.tgt_vocab_size)
        torch.nn.init.xavier_uniform_(self.dense.weight)
        torch.nn.init.xavier_uniform_(self.out.weight)
        self.is_test = False

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
        self.args.graph_tag_num = 2
        self.graph_tagging_embeddings = nn.Embedding(self.args.graph_tag_num, self.args.tagging_embedding_size)
        self.graph_tagging_embeddings.weight.data.uniform_(-0.1, 0.1)

    def encode(self, src_input, answer_tag_input, graph_tag_input, distance_tag_input, src_input_lengths):
        sorted_src_input_length, idx_sort = torch.sort(src_input_lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0, descending=False)
        src_input = src_input[idx_sort]
        answer_tag_input = answer_tag_input[idx_sort]
        graph_tag_input = graph_tag_input[idx_sort]

        src_input_embed = self.dropout(self.src_embedding(src_input))
        answer_tag_input_embed = self.dropout(self.answer_tagging_embeddings(answer_tag_input))
        graph_tag_input_embed = self.dropout(self.graph_tagging_embeddings(graph_tag_input))

        input_embed = torch.cat([src_input_embed, answer_tag_input_embed, graph_tag_input_embed], dim=-1)

        packed_input_embed = nn.utils.rnn.pack_padded_sequence(input_embed, sorted_src_input_length, batch_first=True)
        encoder_outputs, (h_t, c_t) = self.encoder_lstm(packed_input_embed)
        encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(encoder_outputs, batch_first=True)

        encoder_outputs = encoder_outputs[idx_unsort]

        h_t = torch.cat([h_t[0][idx_unsort].unsqueeze(0), h_t[1][idx_unsort].unsqueeze(0)], 0)
        c_t = torch.cat([c_t[0][idx_unsort].unsqueeze(0), c_t[1][idx_unsort].unsqueeze(0)], 0)
        init_h = torch.cat((h_t[0], h_t[1]), dim=-1)
        init_c = torch.cat((c_t[0], c_t[1]), dim=-1)
        init_state = (init_h, init_c)
        return encoder_outputs, init_state

    def decode(self, tgt_input, init_hidden, encoder_outputs, src_input):
        init_context = Variable(encoder_outputs.data.new(tgt_input.size(0), self.args.encoder_hidden_size*2).zero_(),
                                requires_grad=False)

        logits = []
        for t in range(tgt_input.size(1)):
            if t == 0:
                cur_input = tgt_input[:, t]
                cur_context = init_context
            elif not self.is_test:
                cur_input = tgt_input[:, t]

            if not self.args.attn_input_feed:
                input_embed = self.dropout(self.tgt_embedding(cur_input))
            else:
                input_embed = self.dropout(torch.cat([self.tgt_embedding(cur_input), cur_context], -1))

            h_t, c_t = self.decoder_lstm_cell(input_embed, init_hidden)
            h_t = self.dropout(h_t)
            init_hidden = (h_t, c_t)

            alpha_1, cur_context_1 = self.sentence_attn(h_t, encoder_outputs)
            concat_output = torch.cat([h_t.unsqueeze(1), cur_context_1], -1)
            cur_context = cur_context_1.squeeze(1)


            tanh_fc = F.relu(self.dense(concat_output))
            logit = self.out(tanh_fc)

            if self.is_test:
                cur_input = torch.squeeze(torch.max(logit, -1)[1])
            logits.append(logit)

        logits = torch.cat(logits, 1)
        return logits

    def test_decoder(self, cur_input, init_hidden):
        input_embed = self.dropout(self.tgt_embedding(cur_input))
        h_t, c_t = self.decoder_lstm_cell(input_embed, init_hidden)
        h_t = self.dropout(h_t)
        init_hidden = (h_t, c_t)

        if h_t.size(0) == self.args.batch_size:
            encoder_outputs = self.encode_outputs
        else:
            encoder_outputs = torch.cat([self.encode_outputs.unsqueeze(1),]*self.args.beam_size, 1).view(-1, self.encode_outputs.size(1), self.encode_outputs.size(2))

        alpha_1, cur_context_1 = self.sentence_attn(h_t, encoder_outputs)
        concat_output = torch.cat([h_t.unsqueeze(1), cur_context_1], -1)

        tanh_fc = F.relu(self.dense(concat_output))
        logit = F.softmax(self.out(tanh_fc).squeeze(1), -1)

        return logit, init_hidden

    def forward(self, src_input, answer_tag_input, graph_tag_input, distance_tag_input, src_input_lengths, node_input, edge_input, adjency_matrix, answer_input, tgt_input):
        encode_outputs, init_state = self.encode(src_input, answer_tag_input, graph_tag_input, distance_tag_input, src_input_lengths)

        if not self.is_test:
            logits = self.decode(tgt_input, init_state, encode_outputs, src_input)
            return logits
        else:
            if self.args.beam_size > 1:
                # beam search
                self.src_input = src_input
                self.encode_outputs = encode_outputs
                words = self.bs.forward(self.args.max_len, self.test_decoder, tgt_input[:, 0], init_state)
                return words
            else:
                # not beam search
                logits = self.decode(tgt_input, init_state, encode_outputs, src_input)
                return logits


class BeamSearch(object):
    def __init__(self, opts):
        self._options = opts
        self.word_length, self.stops, self.prob = None, None, None
        self.batch_size = None
        self.time = None
        self.prev_index_sequence = None

    def init(self, batch_size):
        self.batch_size = batch_size
        self.word_length = torch.zeros(batch_size).long().cuda()
        self.stops = torch.zeros(batch_size).long().cuda()
        self.prob = torch.ones(batch_size).cuda()
        self.prev_index_sequence = list()

    def forward(self, length, cell, word, state):
        self.init(word.size(0))
        word_list = []
        output_list = []
        for i in range(length):
            self.time = i
            word_prob, next_state = cell(word, state)
            word, state = self.step(next_state, word_prob)
            word_list.append(word)
        word = self.get_output_words(word_list)
        return word

    def get_output_words(self, word_list):
        opts = self._options
        word_sequence = []
        output_sequence = []
        index = torch.arange(self.batch_size).mul(opts.beam_size).long().cuda()
        prev_index_sequence = self.prev_index_sequence
        for word, prev_index in zip(word_list[::-1], prev_index_sequence[::-1]):
            output_word = word.index_select(0, index)
            # word_score = score.index_select(0, index)
            index = prev_index.index_select(0, index)
            word_sequence.append(output_word)
            # score_sequence.append(word_score)
        return torch.stack(word_sequence[::-1], 1)

    def step(self, next_state, word_prob):
        # next_state = cell(word, state)
        # word_prob = F.softmax(cell.end_points.get('word_score'), dim=-1)
        word_prob = self.solve_prob(word_prob)
        word_length = self.solve_length()
        next_word, prev_index = self.solve_score(word_prob, word_length)
        next_state = self.update(prev_index, next_word, next_state, word_prob)
        return next_word, next_state

    def solve_prob(self, word_prob):
        # self.prob: [BatchSize*BeamSize], prob: [BatchSize*BeamSize, VocabSize]
        opts = self._options
        stops = self.stops
        stops = stops.unsqueeze(dim=-1)
        unstop_word_prob = torch.mul(word_prob, (1 - stops).float())
        batch_size = self.batch_size if self.time == 0 else self.batch_size * opts.beam_size
        pad = torch.tensor([[opts.pad_id]]).long().cuda()
        stop_prob = torch.zeros(1, opts.tgt_vocab_size).cuda().scatter_(1, pad, 1.0).repeat(batch_size, 1)
        stop_word_prob = stop_prob.mul(stops.float())
        word_prob = unstop_word_prob.add(stop_word_prob)
        prob = self.prob
        prob = prob.unsqueeze(-1)
        word_prob = prob.mul(word_prob)
        return word_prob

    def solve_length(self):
        # self.word_length: [BatchSize*BeamSize],
        opts, stops, word_length = self._options, self.stops, self.word_length
        stops = stops.unsqueeze(dim=-1)
        word_length = word_length.unsqueeze(dim=-1)
        batch_size = self.batch_size if self.time == 0 else self.batch_size * opts.beam_size
        pad = torch.tensor([[opts.eos_id, opts.pad_id]]).long()
        if torch.cuda.is_available():
            pad = pad.cuda()
        unstop_tokens = torch.ones(1, opts.tgt_vocab_size).cuda().scatter_(1, pad, 0.0).\
            repeat(batch_size, 1).long()
        add_length = unstop_tokens.mul(1 - stops)
        word_length = word_length.add(add_length)
        return word_length

    def solve_score(self, word_prob, word_length):
        opts = self._options
        beam_size = 1 if self.time == 0 else opts.beam_size
        length_penalty = ((word_length + 5).float().pow(opts.length_penalty_factor)).\
            div((torch.tensor([6.0]).cuda()).pow(opts.length_penalty_factor))
        word_score = word_prob.clamp(1e-20, 1.0).log().div(length_penalty)
        # mini = word_score.min()
        word_score = word_score.view(-1, beam_size * opts.tgt_vocab_size)
        beam_score, beam_words = word_score.topk(opts.beam_size)
        prev_index = torch.arange(self.batch_size).long().cuda().mul(beam_size).view(-1, 1).\
            add(beam_words.div(opts.tgt_vocab_size)).view(-1)
        next_words = beam_words.fmod(opts.tgt_vocab_size).view(-1).long()
        self.prev_index_sequence.append(prev_index)
        return next_words, prev_index

    def update(self, index, word, state, prob):
        opts = self._options
        # next_state = []
        # for each_state in state:
        #     next_state.append((each_state[0].index_select(0, index), each_state[1].index_select(0, index)))
        next_state = (state[0].index_select(0, index), state[1].index_select(0, index))
        self.stops = word.le(opts.eos_id).long()
        self.prob = prob.index_select(0, index).gather(1, word.view(-1, 1)).squeeze(1)
        self.word_length = self.word_length.gather(0, index).add(1-self.stops)
        return next_state



