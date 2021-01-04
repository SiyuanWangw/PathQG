import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

from Compared_Models.Answer_Separation_Model import Generator
import sys
sys.path.append('..')
sys.path.append('../..')
sys.path.append('../../..')
from Models.Graph_Models.Compared_Models.SceneGraph_Config import init_args
from SceneGraph_LoadData import *
from Evaluator.METEOR.meteor import Meteor
from Evaluator.ROUGE.rouge import Rouge
from Evaluator.BLEU.bleu import Bleu
bleu = Bleu()
meteor = Meteor()
rouge = Rouge()
import time
start = time.time()
from SceneGraphParser import sng_parser
parser = sng_parser.Parser('spacy', model='en')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Generator_Solver(object):
    def __init__(self, inited_args, vocabulary_src, vocabulary_tgt):
        self.args = inited_args
        self.src_embeddings = np.load(self.args.src_embedding_file)
        self.tgt_embeddings = np.load(self.args.tgt_embedding_file)
        self.vocabulary_src = vocabulary_src
        self.vocabulary_tgt = vocabulary_tgt
        self.word_to_inx_src = dict(zip(vocabulary_src, range(len(vocabulary_src))))

    def adjust_learning_rate(self, optimizer):
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * self.args.lr_decay

    def train(self):
        params = get_graph_train_data(self.args.batch_size, self.vocabulary_src, self.vocabulary_tgt, is_as=True)

        model = Generator(self.args, self.word_to_inx_src['PAD'], self.src_embeddings, self.tgt_embeddings)

        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Number of training parameters:', total_trainable_params)

        if self.args.cuda:
            model = model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=self.args.lr)
        criterion = nn.CrossEntropyLoss(size_average=False, reduce=False)

        evaluator = Test(self.args, self.vocabulary_src, self.vocabulary_tgt, is_test=False)
        best_score = 0
        best_epoch = 0

        for epoch in range(self.args.num_epoches):
            all_vector_src, all_vector_tgt, \
            input_encode_batches, input_encode_lengths, \
            input_nodes_batches, input_edges_batches, input_adj_batches, \
            input_answer_tagging_batches, input_graph_tagging_batches, input_distance_tagging_batches, \
            input_neighbor_tagging_batches, input_decode_batches, _, \
            target_decode_batches, input_answer_batches, input_answer_lengthes = get_graph_train_data_each_epoch(params, is_shuffle=True)

            print('epoch:', epoch, ', learning rate:', optimizer.param_groups[0]['lr'])
            n_chunk = len(all_vector_src) // self.args.batch_size

            for batch in range(n_chunk):
                src_input = Variable(torch.from_numpy(input_encode_batches[batch])).long()
                src_input_length = Variable(torch.from_numpy(input_encode_lengths[batch])).long()
                node_input = torch.from_numpy(input_nodes_batches[batch]).long()
                edge_input = torch.from_numpy(input_edges_batches[batch]).long()
                adj_input = torch.from_numpy(input_adj_batches[batch])
                answer_tagging_input = torch.from_numpy(input_answer_tagging_batches[batch]).long()
                graph_tagging_input = torch.from_numpy(input_graph_tagging_batches[batch]).long()
                distance_tagging_input = torch.from_numpy(input_distance_tagging_batches[batch]).long()
                neighbor_tagging_input = torch.from_numpy(input_neighbor_tagging_batches[batch])
                tgt_input = Variable(torch.from_numpy(input_decode_batches[batch])).long()
                tgt_target = Variable(torch.from_numpy(target_decode_batches[batch])).long()
                answer_input = Variable(torch.from_numpy(input_answer_batches[batch])).long()
                answer_input_length = Variable(torch.from_numpy(input_answer_lengthes[batch])).long()

                if self.args.cuda:
                    src_input = src_input.cuda()
                    src_input_length = src_input_length.cuda()
                    node_input = node_input.cuda()
                    edge_input = edge_input.cuda()
                    adj_input = adj_input.cuda()
                    answer_tagging_input = answer_tagging_input.cuda()
                    graph_tagging_input = graph_tagging_input.cuda()
                    distance_tagging_input = distance_tagging_input.cuda()
                    neighbor_tagging_input = neighbor_tagging_input.cuda()
                    tgt_input = tgt_input.cuda()
                    tgt_target = tgt_target.cuda()
                    answer_input = answer_input.cuda()
                    answer_input_length = answer_input_length.cuda()

                logits = model(src_input, src_input_length, graph_tagging_input, answer_input, answer_input_length, tgt_input)

                qg_loss = criterion(logits.view(-1, logits.shape[2]), tgt_target.view(-1)).view(self.args.batch_size, -1)
                mask = torch.sign(tgt_target).float()
                qg_loss = torch.sum(mask.mul(qg_loss))/torch.sum(mask)

                reg_loss = 0
                for name, param in model.named_parameters():
                    if 'embedding' not in name:
                        reg_loss += self.args.l1_reg * param.abs().sum() + self.args.l2_reg * (param.pow(2)).sum()
                loss = qg_loss + reg_loss

                optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm(model.parameters(), self.args.clip_grad)
                optimizer.step()

                if batch % 50 == 0:
                    print(
                        '[INFO] Epoch: %d , batch: %d , qg loss: %.6f, total loss: %.6f, gradient norm: %.6f' % (
                        epoch, batch, qg_loss.item(), loss.item(), grad_norm))

            self.adjust_learning_rate(optimizer)
            model.eval()
            bleu_score, meteor_score, rouge_score = evaluator.evaluate(model)
            if bleu_score[0] > best_score:
                best_score = bleu_score[0]
                best_epoch = epoch
                torch.save(model, self.args.QG_model_file)
                print('model saved!!!!!!!!!')
            print('current best score:', best_score)
            print('current best epoch:', best_epoch)
            model.train()


class Test(object):
    def __init__(self, inited_args, vocabulary_src, vocabulary_tgt, is_test=False):
        self.args = inited_args
        self.src_embeddings = np.load(self.args.src_embedding_file)
        self.tgt_embeddings = np.load(self.args.tgt_embedding_file)
        self.vocabulary_src = vocabulary_src
        self.vocabulary_tgt = vocabulary_tgt
        self.word_to_inx_src = dict(zip(vocabulary_src, range(len(vocabulary_src))))

        self.is_test = is_test

        # load data
        if is_test is False:
            all_vector_src, all_vector_tgt, \
            self.input_encode_batches, self.input_encode_lengths, \
            self.input_nodes_batches, self.input_edges_batches, self.input_adj_batches, \
            self.input_answer_tagging_batches, self.input_graph_tagging_batches, self.input_distance_tagging_batches, \
            self.input_neighbor_tagging_batches, self.input_decode_batches, self.all_questions, \
            self.input_answer_batches, self.input_answer_lengthes = get_graph_val_data(self.args.batch_size, self.vocabulary_src,
                                                              self.vocabulary_tgt, is_as=True)
            self.long_data_indexes = np.load(
                os.path.join(dataroot,
                             './processed/SQuAD1.0/Graph_Analysis/SceneGraph/val/long_data_piece_index.npy')).tolist()
        else:
            all_vector_src, all_vector_tgt, \
            self.input_encode_batches, self.input_encode_lengths, \
            self.input_nodes_batches, self.input_edges_batches, self.input_adj_batches, \
            self.input_answer_tagging_batches, self.input_graph_tagging_batches, self.input_distance_tagging_batches, \
            self.input_neighbor_tagging_batches, self.input_decode_batches, self.all_questions, \
            self.input_answer_batches, self.input_answer_lengthes, self.all_sentences, self.spread_spice_ref_questions = get_graph_test_data(
                self.args.batch_size, self.vocabulary_src, self.vocabulary_tgt, is_as=True)
            self.long_data_indexes = np.load(
                os.path.join(dataroot,
                             './processed/SQuAD1.0/Graph_Analysis/SceneGraph/test/long_data_piece_index.npy')).tolist()

    def compute_overlap(self, gene_q, ref_s):
        gene_q_tokens = nltk.word_tokenize(gene_q)
        ref_tokens = nltk.word_tokenize(ref_s)
        overlap_num = 0
        for each_word in gene_q_tokens:
            if each_word in ref_tokens:
                overlap_num += 1
        if len(gene_q_tokens) == 0:
            return 0
        else:
            return overlap_num/len(gene_q_tokens)

    def compute_BLEU_Meteor_ROUGE(self, gene_questions, val_questions, ref_sentences=None, long_result=True):
        ref_dict = {}
        gene_dict = {}
        generate_questions_length = len(gene_questions)
        overlap_rates = []

        print(len(gene_questions), len(val_questions))
        for index in range(generate_questions_length):
            if long_result:
                if index in self.long_data_indexes: # long data
                    generated_question = gene_questions[index]
                    gene_dict[str(index + 1)] = [generated_question]
                    reference_questions = val_questions[index]  # val_questions[src_dev[index]]
                    reference_questions_list = [' '.join(each.split()[1:-1]) for each in reference_questions]
                    ref_dict[str(index + 1)] = reference_questions_list
                    if ref_sentences is not None:
                        overlap_rates.append(self.compute_overlap(gene_questions[index], ref_sentences[index]))
            else:
                generated_question = gene_questions[index]
                gene_dict[str(index + 1)] = [generated_question]
                reference_questions = val_questions[index]  # val_questions[src_dev[index]]
                reference_questions_list = [' '.join(each.split()[1:-1]) for each in reference_questions]
                ref_dict[str(index + 1)] = reference_questions_list
                if ref_sentences is not None:
                    overlap_rates.append(self.compute_overlap(gene_questions[index], ref_sentences[index]))

        bleu_score, all_bleu_scores = bleu.compute_score(ref_dict, gene_dict)
        meteor_score, all_meteor_scores = meteor.compute_score(ref_dict, gene_dict)
        rouge_score, all_rouge_scores = rouge.compute_score(ref_dict, gene_dict)
        print('The BLEU score is:')
        print(bleu_score)
        print('The METEOR score is:', meteor_score)
        print('The ROUGE score is:', rouge_score)

        if ref_sentences is not None:
            final_overlap_rate = np.mean(overlap_rates)
            print('overlap rate', final_overlap_rate)
        return bleu_score, meteor_score, rouge_score, all_bleu_scores, all_meteor_scores, all_rouge_scores

    def compute_SPICE(self, gene_questions, ref_questions, long_result=True):
        total_f1_score = 0
        long_num = 0
        for index in tqdm(range(len(gene_questions))):
            if long_result:
                if index in self.long_data_indexes: # long data
                    gene_graph_dict = parser.parse(gene_questions[index])
                    gene_semantic_propositions = self.get_semantic_propositions(gene_graph_dict)
                    ref_graph_dict = parser.parse(' '.join(ref_questions[index].split()[1:-1]) )
                    ref_semantic_propositions = self.get_semantic_propositions(ref_graph_dict)

                    total_f1_score += self.f1(gene_semantic_propositions, ref_semantic_propositions)
                    long_num += 1
            else:
                gene_graph_dict = parser.parse(gene_questions[index])
                gene_semantic_propositions = self.get_semantic_propositions(gene_graph_dict)
                ref_graph_dict = parser.parse(' '.join(ref_questions[index].split()[1:-1]) )
                ref_semantic_propositions = self.get_semantic_propositions(ref_graph_dict)

                total_f1_score += self.f1(gene_semantic_propositions, ref_semantic_propositions)
                long_num += 1

        final_f1_score = total_f1_score/long_num
        print('SPICE', final_f1_score)
        return final_f1_score

    def get_semantic_propositions(self, graph_dict):
        semantic_propositions = list()
        for each_entity in graph_dict['entities']:
            semantic_propositions.append(each_entity['span'])
            semantic_propositions.append(each_entity['head'])
            if len(each_entity['modifiers']) > 0:
                for each_modifier in each_entity['modifiers']:
                    # if each_modifier['span'] not in det_words:
                    semantic_propositions.append(each_modifier['span'])
        for each_relation in graph_dict['relations']:
            semantic_propositions.append(each_relation['relation'])
        return set(semantic_propositions)

    def f1(self, candidate, reference):
        epsilon = 1e-7
        matching_number = len(reference & candidate)
        p = matching_number / (len(candidate) + epsilon)
        r = matching_number / (len(reference) + epsilon)

        f1 = 2 * p * r / (p + r + epsilon)

        return f1

    def evaluate(self, model=None):
        if model is None:
            old_model = torch.load(self.args.QG_model_file)
            model = Generator(self.args, self.word_to_inx_src['PAD'], self.src_embeddings, self.tgt_embeddings)
            model.load_state_dict(old_model.state_dict())
            model.eval()
        if self.args.cuda:
            model = model.cuda()
        model.is_test = True

        gene_questions = list()
        n_chunk = len(self.input_encode_batches)

        for batch in range(n_chunk):
            src_input = Variable(torch.from_numpy(self.input_encode_batches[batch])).long()
            src_input_length = Variable(torch.from_numpy(self.input_encode_lengths[batch])).long()
            node_input = Variable(torch.from_numpy(self.input_nodes_batches[batch])).long()
            edge_input = Variable(torch.from_numpy(self.input_edges_batches[batch])).long()
            adj_input = Variable(torch.from_numpy(self.input_adj_batches[batch]))
            answer_tagging_input = Variable(torch.from_numpy(self.input_answer_tagging_batches[batch])).long()
            graph_tagging_input = Variable(torch.from_numpy(self.input_graph_tagging_batches[batch])).long()
            distance_tagging_input = Variable(torch.from_numpy(self.input_distance_tagging_batches[batch])).long()
            neighbor_tagging_input = Variable(torch.from_numpy(self.input_neighbor_tagging_batches[batch]))
            tgt_input = Variable(torch.from_numpy(self.input_decode_batches[batch])).long()
            answer_input = Variable(torch.from_numpy(self.input_answer_batches[batch])).long()
            answer_input_length = Variable(torch.from_numpy(self.input_answer_lengthes[batch])).long()

            if self.args.cuda:
                src_input = src_input.cuda()
                src_input_length = src_input_length.cuda()
                node_input = node_input.cuda()
                edge_input = edge_input.cuda()
                adj_input = adj_input.cuda()
                answer_tagging_input = answer_tagging_input.cuda()
                graph_tagging_input = graph_tagging_input.cuda()
                distance_tagging_input = distance_tagging_input.cuda()
                neighbor_tagging_input = neighbor_tagging_input.cuda()
                tgt_input = tgt_input.cuda()
                answer_input = answer_input.cuda()
                answer_input_length = answer_input_length.cuda()

            if self.args.beam_size > 1:
                # beam search
                pre_index = model(src_input, src_input_length, graph_tagging_input, answer_input, answer_input_length, tgt_input)
            else:
                logits = model(src_input, src_input_length, graph_tagging_input, answer_input, answer_input_length, tgt_input)
                pre_index = torch.max(logits, -1)[1]

            for i in range(pre_index.size(0)):
                generate_string = indexs2str(pre_index[i], self.vocabulary_tgt, is_question=True)
                gene_questions.append(generate_string)

            if batch % 20 == 0:
                print('batch', batch)

        if self.is_test:
            bleu_score, meteor_score, rouge_score, all_bleu_scores, all_meteor_scores, all_rouge_scores = self.compute_BLEU_Meteor_ROUGE(
                gene_questions, self.all_questions, self.all_sentences, long_result=self.args.evaluate_complex_data)

            self.compute_SPICE(gene_questions, self.spread_spice_ref_questions, long_result=self.args.evaluate_complex_data)
        else:
            bleu_score, meteor_score, rouge_score, all_bleu_scores, all_meteor_scores, all_rouge_scores = self.compute_BLEU_Meteor_ROUGE(
                gene_questions, self.all_questions, long_result=self.args.evaluate_complex_data)

        model.is_test = False
        return bleu_score, meteor_score, rouge_score


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    inited_args = init_args()

    vocabulary_src = np.load(
        os.path.join(dataroot, './processed/SQuAD1.0/Graph_Analysis/SceneGraph/graph_vocabulary_src.npy'))
    vocabulary_tgt = np.load(
        os.path.join(dataroot, './processed/SQuAD1.0/Graph_Analysis/SceneGraph/graph_vocabulary_tgt.npy'))
    print('vocabulary loaded...')
    print('The size of src and tgt vocabulary are', len(vocabulary_src), len(vocabulary_tgt))

    inited_args.src_vocab_size = len(vocabulary_src)
    inited_args.tgt_vocab_size = len(vocabulary_tgt)

    # trainer = Generator_Solver(inited_args, vocabulary_src, vocabulary_tgt)
    evaluator = Test(inited_args, vocabulary_src, vocabulary_tgt, is_test=True)

    # trainer.train()
    evaluator.evaluate()

    print(evaluator.args.QG_model_file)
    print('running time', time.time()-start)