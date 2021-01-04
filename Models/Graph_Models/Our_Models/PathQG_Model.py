import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
sys.path.append('..')
sys.path.append('../..')
from Attention import BahdanauAttention


class Generator(nn.Module):
    def __init__(self, args, pad_idx=0, pretrain_src_embeddings=None, pretrain_tgt_embeddings=None):
        super(Generator, self).__init__()
        self.args = args
        self.bs = BeamSearch(self.args)
        self.init_embedding(pad_idx, pretrain_src_embeddings, pretrain_tgt_embeddings)
        self.src_encoding_lstm = nn.LSTM(self.args.embedding_size + self.args.tagging_embedding_size * 2,
                                    self.args.encoder_hidden_size,
                                    num_layers=self.args.num_layers,
                                    bidirectional=True,
                                    batch_first=True)

        self.path_encoder_forward_lstm_cell =nn.LSTMCell(self.args.embedding_size,
                                         self.args.path_hidden_size)
        self.path_encoder_backward_lstm_cell = nn.LSTMCell(self.args.embedding_size,
                                         self.args.path_hidden_size)

        self.encoder_forward_lstm_cell = nn.LSTMCell(self.args.embedding_size,
                                             self.args.entity_encoder_hidden_size)
        self.encoder_backward_lstm_cell = nn.LSTMCell(self.args.embedding_size,
                                             self.args.entity_encoder_hidden_size)

        self.decoder_lstm_cell = nn.LSTMCell(self.args.embedding_size + self.args.entity_dense_size*2,
                                             self.args.decoder_hidden_size)

        self.sentence_attn = BahdanauAttention(self.args.entity_encoder_hidden_size*2, self.args.encoder_hidden_size * 2,
                                               self.args.attn_size)
        self.decoder_sentence_attn = BahdanauAttention(self.args.decoder_hidden_size, self.args.encoder_hidden_size * 2,
                                               self.args.attn_size)
        self.decoder_entity_attn = BahdanauAttention(self.args.decoder_hidden_size, self.args.path_hidden_size*2,
                                                     self.args.attn_size)
        self.dropout = nn.Dropout(self.args.dropout)

        self.entity_dense = nn.Linear(self.args.entity_encoder_hidden_size * 2 + self.args.encoder_hidden_size * 2,
                               self.args.entity_dense_size)
        self.entity_out = nn.Linear(self.args.entity_dense_size, self.args.entity_tag_num)
        torch.nn.init.xavier_uniform_(self.entity_dense.weight)
        torch.nn.init.xavier_uniform_(self.entity_out.weight)

        self.dense = nn.Linear(self.args.decoder_hidden_size + self.args.encoder_hidden_size * 2 + self.args.path_hidden_size*2,
                                self.args.question_dense_size)
        self.out = nn.Linear(self.args.question_dense_size, self.args.tgt_vocab_size)
        self.skip_project = nn.Linear(self.args.path_hidden_size+self.args.embedding_size, self.args.path_hidden_size)
        torch.nn.init.xavier_uniform_(self.dense.weight)
        torch.nn.init.xavier_uniform_(self.out.weight)
        torch.nn.init.xavier_uniform_(self.skip_project.weight)

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
        self.end_tagging_embeddings = nn.Embedding(self.args.end_tag_num, self.args.tagging_embedding_size)
        self.end_tagging_embeddings.weight.data.uniform_(-0.1, 0.1)

    def src_encoding(self, src_input, src_input_lengths, answer_tag_input, end_tag_input, lstm_entity_embedding=None):
        # encoding of src input
        sorted_src_input_length, idx_sort = torch.sort(src_input_lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0, descending=False)
        src_input = src_input[idx_sort]
        answer_tag_input = answer_tag_input[idx_sort]
        end_tag_input = end_tag_input[idx_sort]

        src_input_embeddings = self.dropout(self.src_embedding(src_input))
        answer_tag_input_embeddings = self.dropout(self.answer_tagging_embeddings(answer_tag_input))
        end_tag_input_embeddings = self.dropout(self.end_tagging_embeddings(end_tag_input))

        if lstm_entity_embedding is not None:
            h_contexts = []
            for t in range(src_input_embeddings.size(1)):
                _, h_context = self.init_attn(src_input_embeddings[:, t, :], lstm_entity_embedding)
                h_contexts.append(h_context)
            h_contexts = torch.cat(h_contexts, 1)
            input_embeddings = torch.cat([src_input_embeddings, answer_tag_input_embeddings, end_tag_input_embeddings, h_contexts],
                                         dim=-1)
        else:
            input_embeddings = torch.cat([src_input_embeddings, answer_tag_input_embeddings, end_tag_input_embeddings],
                                     dim=-1)

        packed_input_embeddings = nn.utils.rnn.pack_padded_sequence(input_embeddings, sorted_src_input_length,
                                                                    batch_first=True)
        encoder_outputs, (h_t, c_t) = self.src_encoding_lstm(packed_input_embeddings)
        encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(encoder_outputs, batch_first=True)

        encoder_outputs = encoder_outputs[idx_unsort]

        h_t = torch.cat([h_t[0][idx_unsort].unsqueeze(0), h_t[1][idx_unsort].unsqueeze(0)], 0)
        c_t = torch.cat([c_t[0][idx_unsort].unsqueeze(0), c_t[1][idx_unsort].unsqueeze(0)], 0)
        init_h = torch.cat((h_t[0], h_t[1]), dim=-1)
        init_c = torch.cat((c_t[0], c_t[1]), dim=-1)
        init_state = (init_h, init_c)

        return encoder_outputs, init_state

    def prior_encoder(self, encoder_outputs, average_entity_embedding):
        # entity_seq_input: b*N*l
        fw_outputs = []
        bw_outputs = []
        for t in range(average_entity_embedding.size(1)):
            fw_cur_input = average_entity_embedding[:, t]
            fw_cur_input = self.dropout(fw_cur_input)

            bw_cur_input = average_entity_embedding[:, -(t+1)]
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

        dense_outputs = F.relu(self.entity_dense(outputs))
        logits = self.entity_out(dense_outputs)

        return logits

    def decoder(self, tgt_input, init_hidden, encoder_outputs, posterior_L):
        init_context = Variable(encoder_outputs.data.new(tgt_input.size(0), self.args.entity_dense_size*2).zero_(),
                                requires_grad=False)

        logits = []
        outputs = []
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

            alpha_1, cur_context_1 = self.decoder_sentence_attn(h_t, encoder_outputs)
            alpha_2, cur_context_2 = self.decoder_entity_attn(h_t, posterior_L)
            concat_output = torch.cat([h_t.unsqueeze(1), cur_context_1, cur_context_2], -1)
            outputs.append(concat_output)
            cur_context = torch.cat([cur_context_1, cur_context_2], -1).squeeze(1)
            # concat output size: 64*1*1500

            tanh_fc = F.relu(self.dense(concat_output))
            logit = self.out(tanh_fc)

            if self.is_test:
                cur_input = torch.squeeze(torch.max(logit, -1)[1])
            logits.append(logit)

        logits = torch.cat(logits, 1)
        outputs = torch.cat(outputs, 1)

        return logits, outputs

    def test_decoder(self, cur_input, init_hidden, t):
        if cur_input.size(0) == self.args.batch_size:
            encoder_outputs = self.encoder_outputs
            prior_L = self.prior_L
        else:
            encoder_outputs = torch.cat([self.encoder_outputs.unsqueeze(1),]*self.args.beam_size, 1).view(-1, self.encoder_outputs.size(1), self.encoder_outputs.size(2))
            prior_L = torch.cat([self.prior_L.unsqueeze(1),]*self.args.beam_size, 1).view(-1, self.prior_L.size(1), self.prior_L.size(2))

        if t == 0:
            cur_context = Variable(
                init_hidden[0].data.new(cur_input.size(0), self.args.entity_dense_size*2).zero_(),
                requires_grad=False)
        else:
            _, cur_context_1 = self.decoder_sentence_attn(init_hidden[0], encoder_outputs)
            _, cur_context_2 = self.decoder_entity_attn(init_hidden[0], prior_L)
            cur_context = torch.cat([cur_context_1, cur_context_2], -1).squeeze(1)

        if not self.args.attn_input_feed:
            input_embed = self.dropout(self.tgt_embedding(cur_input))
        else:
            input_embed = self.dropout(torch.cat([self.tgt_embedding(cur_input), cur_context], -1))

        h_t, c_t = self.decoder_lstm_cell(input_embed, init_hidden)
        h_t = self.dropout(h_t)
        init_hidden = (h_t, c_t)

        alpha_1, cur_context_1 = self.decoder_sentence_attn(h_t, encoder_outputs)
        _, cur_context_2 = self.decoder_entity_attn(h_t, prior_L)

        concat_output = torch.cat([h_t.unsqueeze(1), cur_context_1, cur_context_2], -1)

        tanh_fc = F.relu(self.dense(concat_output))
        logit = F.softmax(self.out(tanh_fc).squeeze(1), -1)

        return logit, init_hidden, concat_output.squeeze(1)

    def encoding_path(self, logits, node_embedding):
        node_embedding = torch.mul(F.softmax(logits, dim=-1)[:, :, 0].unsqueeze(-1), node_embedding)
        input_embedding = node_embedding

        fw_outputs = []
        bw_outputs = []
        for t in range(input_embedding.size(1)):
            fw_cur_input = input_embedding[:, t]
            fw_cur_input = self.dropout(fw_cur_input)
            bw_cur_input = input_embedding[:, -(t + 1)]
            bw_cur_input = self.dropout(bw_cur_input)

            if t == 0:
                fw_h_t, fw_c_t = self.path_encoder_forward_lstm_cell(fw_cur_input)
                bw_h_t, bw_c_t = self.path_encoder_backward_lstm_cell(bw_cur_input)
            else:
                fw_h_t, fw_c_t = self.path_encoder_forward_lstm_cell(fw_cur_input, fw_init_hidden)
                bw_h_t, bw_c_t = self.path_encoder_backward_lstm_cell(bw_cur_input, bw_init_hidden)

            fw_init_hidden = (fw_h_t, fw_c_t)
            bw_init_hidden = (bw_h_t, bw_c_t)

            if t % 2 == 0:
                fw_outputs.append(fw_h_t.unsqueeze(1))
                bw_outputs.append(bw_h_t.unsqueeze(1))
            else:
                fw_h_t_skip = self.skip_project(torch.cat([fw_h_t, input_embedding[:, t-1]], -1))
                bw_h_t_skip = self.skip_project(torch.cat([bw_h_t, input_embedding[:, t+1]], -1))

                fw_outputs.append(fw_h_t_skip.unsqueeze(1))
                bw_outputs.append(bw_h_t_skip.unsqueeze(1))

            # fw_outputs.append(fw_h_t.unsqueeze(1))
            # bw_outputs.append(bw_h_t.unsqueeze(1))


        init_h = torch.cat([fw_h_t, bw_h_t], -1)
        init_c = torch.cat([fw_c_t, bw_c_t], -1)

        bw_outputs.reverse()
        fw_outputs = torch.cat(fw_outputs, 1)
        bw_outputs = torch.cat(bw_outputs, 1)
        outputs = torch.cat([fw_outputs, bw_outputs], -1)

        return outputs, (init_h, init_c)

    def forward(self, src_input, src_input_lengths, answer_tag_input, end_tag_input, entity_seq_input, entity_type_seq_input, edge_seq_input, tgt_input, GT_prob):

        mask = torch.sign(entity_seq_input).float().unsqueeze(-1)
        entity_embedding = self.src_embedding(entity_seq_input)
        average_entity_embedding = torch.sum(mask.mul(entity_embedding), 2) / torch.sum(mask, 2).clamp(1)

        encoder_outputs, init_state = self.src_encoding(src_input, src_input_lengths, answer_tag_input, end_tag_input)

        prior_logits = self.prior_encoder(encoder_outputs, average_entity_embedding)

        self.prior_L, init_path_states = self.encoding_path(prior_logits, average_entity_embedding)

        final_init_h = torch.cat([init_state[0], init_path_states[0]], -1)
        final_init_c = torch.cat([init_state[1], init_path_states[1]], -1)
        final_init_state = (final_init_h, final_init_c)

        if not self.is_test:
            logits, question_outputs = self.decoder(tgt_input, final_init_state, encoder_outputs, self.prior_L)

            return logits, prior_logits
        else:
            # beam search
            if self.args.beam_size > 1:
                self.src_input = src_input
                self.encoder_outputs = encoder_outputs
                words, question_outputs = self.bs.forward(self.args.max_len, self.test_decoder, tgt_input[:, 0], final_init_state)
                return words, prior_logits
            else:
                # not beam search
                logits, question_outputs = self.decoder(tgt_input, final_init_state, encoder_outputs, self.prior_L)
                return logits, prior_logits


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
            word_prob, next_state, output = cell(word, state, self.time)
            word, state, output = self.step(next_state, word_prob, output)
            word_list.append(word)
            output_list.append(output)
        word, outputs = self.get_output_words(word_list, output_list)
        return word, outputs

    def get_output_words(self, word_list, output_list):
        opts = self._options
        word_sequence = []
        output_sequence = []
        index = torch.arange(self.batch_size).mul(opts.beam_size).long().cuda()
        prev_index_sequence = self.prev_index_sequence
        for word, output, prev_index in zip(word_list[::-1], output_list[::-1], prev_index_sequence[::-1]):
            output_word = word.index_select(0, index)
            # word_score = score.index_select(0, index)
            output_s = output.index_select(0, index)
            index = prev_index.index_select(0, index)
            word_sequence.append(output_word)
            output_sequence.append(output_s)
            # score_sequence.append(word_score)
        return torch.stack(word_sequence[::-1], 1), torch.stack(output_sequence[::-1], 1)

    def step(self, next_state, word_prob, output):
        # next_state = cell(word, state)
        # word_prob = F.softmax(cell.end_points.get('word_score'), dim=-1)
        word_prob = self.solve_prob(word_prob)
        word_length = self.solve_length()
        next_word, prev_index = self.solve_score(word_prob, word_length)
        next_state, next_output = self.update(prev_index, next_word, next_state, word_prob, output)
        return next_word, next_state, next_output

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

    def update(self, index, word, state, prob, output):
        opts = self._options
        next_state = (state[0].index_select(0, index), state[1].index_select(0, index))
        next_output = output.index_select(0, index)
        self.stops = word.le(opts.eos_id).long()
        self.prob = prob.index_select(0, index).gather(1, word.view(-1, 1)).squeeze(1)
        self.word_length = self.word_length.gather(0, index).add(1-self.stops)
        return next_state, next_output




