import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from Our_Models.PathQG_Model import Generator  # PathQG
import sys
sys.path.append('..')
sys.path.append('../..')
from Our_Models.PathQG_Config import init_args
from SceneGraph_LoadData import *
sys.path.append('../../..')
from Evaluator.METEOR.meteor import Meteor
from Evaluator.ROUGE.rouge import Rouge
from Evaluator.BLEU.bleu import Bleu
bleu = Bleu()
meteor = Meteor()
rouge = Rouge()
from sklearn.metrics import f1_score
import numpy as np
from SceneGraphParser import sng_parser
parser = sng_parser.Parser('spacy', model='en')


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
        params = get_path_train_data(self.args.batch_size, self.vocabulary_src, self.vocabulary_tgt)

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

        beta = 0.1

        for epoch in range(self.args.num_epoches):
            # load data
            all_vector_src, all_vector_tgt, \
            input_encode_batches, input_encode_lengths, \
            input_nodes_batches, input_edges_batches, _, \
            input_answer_tagging_batches, input_end_tagging_batches, \
            input_path_tagging_batches, _, valid_node_label_batches, input_decode_batches, \
            _, target_decode_batches, input_decode_lengths, input_nodes_type_batches, \
            _, _, nodes_edges_subpaths_batches = get_path_train_data_each_epoch(params, is_shuffle=True)

            print('epoch:', epoch, ', learning rate:', optimizer.param_groups[0]['lr'])
            n_chunk = len(all_vector_src) // self.args.batch_size

            for batch in range(n_chunk):
                src_input = Variable(torch.from_numpy(input_encode_batches[batch])).long()
                src_input_length = Variable(torch.from_numpy(input_encode_lengths[batch])).long()
                node_input = Variable(torch.from_numpy(input_nodes_batches[batch])).long()
                node_type_input = Variable(torch.from_numpy(input_nodes_type_batches[batch])).long()
                edge_input = Variable(torch.from_numpy(input_edges_batches[batch])).long()
                answer_tagging_input = Variable(torch.from_numpy(input_answer_tagging_batches[batch])).long()
                end_tagging_input = Variable(torch.from_numpy(input_end_tagging_batches[batch])).long()
                tgt_input = Variable(torch.from_numpy(input_decode_batches[batch])).long()
                tgt_target = Variable(torch.from_numpy(target_decode_batches[batch])).long()
                tgt_input_length = Variable(torch.from_numpy(input_decode_lengths[batch])).long()
                target_valid_node_label = Variable(torch.from_numpy(valid_node_label_batches[batch])).long()

                if self.args.cuda:
                    src_input = src_input.cuda()
                    src_input_length = src_input_length.cuda()
                    node_input = node_input.cuda()
                    node_type_input = node_type_input.cuda()
                    edge_input = edge_input.cuda()
                    answer_tagging_input = answer_tagging_input.cuda()
                    end_tagging_input = end_tagging_input.cuda()
                    tgt_input = tgt_input.cuda()
                    tgt_target = tgt_target.cuda()
                    target_valid_node_label = target_valid_node_label.cuda()
                    tgt_input_length = tgt_input_length.cuda()

                # PathQG Model
                logits, selection_logits_1 = model(src_input, src_input_length, answer_tagging_input,
                                                    end_tagging_input, node_input,
                                                    node_type_input, edge_input,
                                                    tgt_input, tgt_input_length)

                qg_loss = criterion(logits.view(-1, logits.shape[2]), tgt_target.view(-1)).view(self.args.batch_size, -1)
                qg_mask = torch.sign(tgt_target).float()
                qg_loss = torch.sum(qg_mask.mul(qg_loss)) / torch.sum(qg_mask)

                selection_loss_1 = criterion(selection_logits_1.view(-1, selection_logits_1.shape[2]), target_valid_node_label.view(-1)).view(
                    self.args.batch_size, -1)
                selection_mask = torch.sign(torch.sum(node_input, -1)).float()
                selection_loss_1 = torch.sum(selection_mask.mul(selection_loss_1)) / torch.sum(selection_mask)

                reg_loss = 0
                for name, param in model.named_parameters():
                    if 'embedding' not in name:
                        reg_loss += self.args.l1_reg * param.abs().sum() + self.args.l2_reg * (param.pow(2)).sum()

                loss = qg_loss + beta * selection_loss_1 + reg_loss

                optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm(model.parameters(), self.args.clip_grad)
                optimizer.step()

                if batch % 50 == 0:
                    print(
                        '[INFO] Epoch: %d , batch: %d , qg loss: %.6f, Selection_loss: %.6f, total loss: %.6f, gradient norm: %.6f' % (
                            epoch, batch, qg_loss.item(), selection_loss_1.item(), loss.item(), grad_norm))
                    if torch.isnan(qg_loss).item():
                        return

            self.adjust_learning_rate(optimizer)
            model.eval()
            bleu_score, meteor_score, rouge_score, accuracy = evaluator.evaluate(model)
            if bleu_score[0] > best_score:
                best_score = bleu_score[0]
                best_epoch = epoch
                torch.save(model.state_dict(), self.args.QG_model_file)
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
            self.input_nodes_batches, self.input_edges_batches, _, \
            self.input_answer_tagging_batches, self.input_end_tagging_batches, \
            self.input_path_tagging_batches, _, self.valid_node_label_batches, self.input_decode_batches, \
            self.all_questions, self.input_decode_lengths, self.input_nodes_type_batches, \
            _, _, self.nodes_edges_subpaths_batches = get_path_val_data(self.args.batch_size, self.vocabulary_src, self.vocabulary_tgt)
            self.long_data_indexes = np.load(
                os.path.join(dataroot,
                             './processed/SQuAD1.0/Graph_Analysis/SceneGraph/val/long_data_piece_index.npy')).tolist()

        else:
            all_vector_src, all_vector_tgt, \
            self.input_encode_batches, self.input_encode_lengths, \
            self.input_nodes_batches, self.input_edges_batches, _, \
            self.input_answer_tagging_batches, self.input_end_tagging_batches, \
            self.input_path_tagging_batches, _, self.valid_node_label_batches, self.input_decode_batches, \
            self.all_questions, self.input_decode_lengths, self.input_nodes_type_batches, \
            _, self.spread_spice_ref_questions, self.all_sentences, self.nodes_edges_subpaths_batches = get_path_test_data(self.args.batch_size, self.vocabulary_src, self.vocabulary_tgt)
            self.file = os.path.join(dataroot, './processed/SQuAD1.0/Graph_Analysis/SceneGraph/test/selected_path_extend_3.npy')
            self.long_data_indexes = np.load(
                os.path.join(dataroot,
                             './processed/SQuAD1.0/Graph_Analysis/SceneGraph/test/long_data_piece_index.npy')).tolist()

    def compute_BLEU_Meteor_ROUGE(self, gene_questions, val_questions, training_questions=None, long_result=True):
        ref_dict = {}
        gene_dict = {}
        generate_questions_length = len(gene_questions)
        print(len(gene_questions), len(val_questions))
        overlap_rates = []

        for index in tqdm(range(generate_questions_length)):
            if long_result:
                if index in self.long_data_indexes: # long data
                    generated_question = gene_questions[index]
                    gene_dict[str(index + 1)] = [generated_question]
                    reference_questions = val_questions[index]  # val_questions[src_dev[index]]
                    reference_questions_list = [' '.join(each.split()[1:-1]) for each in reference_questions]
                    ref_dict[str(index + 1)] = reference_questions_list
                    if training_questions is not None:
                        overlap_rates.append(self.compute_overlap_sent(gene_questions[index], training_questions[index]))
            else:
                generated_question = gene_questions[index]
                gene_dict[str(index + 1)] = [generated_question]
                reference_questions = val_questions[index]  # val_questions[src_dev[index]]
                reference_questions_list = [' '.join(each.split()[1:-1]) for each in reference_questions]
                ref_dict[str(index + 1)] = reference_questions_list
                if training_questions is not None:
                    overlap_rates.append(self.compute_overlap_sent(gene_questions[index], training_questions[index]))

        bleu_score, all_bleu_scores = bleu.compute_score(ref_dict, gene_dict)
        meteor_score, all_meteor_scores = meteor.compute_score(ref_dict, gene_dict)
        rouge_score, all_rouge_scores = rouge.compute_score(ref_dict, gene_dict)
        print('The BLEU score is:')
        print(bleu_score)
        print('The METEOR score is:', meteor_score)
        print('The ROUGE score is:', rouge_score)

        if training_questions is not None:
            average_overlap_rate = np.mean(overlap_rates)
            print('average overlap rate', np.round(average_overlap_rate, decimals=4))

        return bleu_score, meteor_score, rouge_score, all_bleu_scores, all_meteor_scores, all_rouge_scores

    def compute_overlap_sent(self, gene_q, ref_s):
        gene_q_tokens = nltk.word_tokenize(gene_q)
        ref_tokens = nltk.word_tokenize(ref_s)
        overlap_num = 0
        for each_word in gene_q_tokens:
            if each_word in ref_tokens:
                overlap_num += 1
        return overlap_num/len(gene_q_tokens)

    def compute_accuarcy(self, reference, prediction):
        correct_indices = np.where(reference>0)[0]
        predict_indices = np.where(prediction>0)[0]
        predict_correct_number = 0
        for each in correct_indices:
            if each in predict_indices:
                predict_correct_number += 1
        if len(correct_indices) > 0:
            return predict_correct_number/len(correct_indices)
        else:
            return 0

    def compute_SPICE(self, gene_questions, ref_questions, long_result=True):
        total_f1_score = 0
        total_num = 0
        all_f1_score = list()
        for index in tqdm(range(len(gene_questions))):
            if long_result:
                if index in self.long_data_indexes:  # long data
                    gene_graph_dict = parser.parse(gene_questions[index])
                    gene_semantic_propositions = self.get_semantic_propositions(gene_graph_dict)
                    ref_graph_dict = parser.parse(' '.join(ref_questions[index].split()[1:-1]) )
                    ref_semantic_propositions = self.get_semantic_propositions(ref_graph_dict)

                    cur_f1 = self.f1(gene_semantic_propositions, ref_semantic_propositions)
                    total_f1_score += cur_f1
                    all_f1_score.append(cur_f1)
                    total_num += 1
            else:
                gene_graph_dict = parser.parse(gene_questions[index])
                gene_semantic_propositions = self.get_semantic_propositions(gene_graph_dict)
                ref_graph_dict = parser.parse(' '.join(ref_questions[index].split()[1:-1]))
                ref_semantic_propositions = self.get_semantic_propositions(ref_graph_dict)

                cur_f1 = self.f1(gene_semantic_propositions, ref_semantic_propositions)
                total_f1_score += cur_f1
                all_f1_score.append(cur_f1)
                total_num += 1

        final_f1_score = total_f1_score/total_num
        print('SPICE', final_f1_score)
        return final_f1_score, all_f1_score

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
            model.load_state_dict(old_model)
            model.eval()
        if self.args.cuda:
            model = model.cuda()
        model.is_test = True

        gene_questions = list()
        n_chunk = len(self.input_encode_batches)
        total_f1_score = 0
        total_accuracy = 0
        selected_paths = list()

        ith = 0
        total_num = 0
        for batch in range(n_chunk):
            src_input = Variable(torch.from_numpy(self.input_encode_batches[batch])).long()
            src_input_length = Variable(torch.from_numpy(self.input_encode_lengths[batch])).long()
            node_input = Variable(torch.from_numpy(self.input_nodes_batches[batch])).long()
            node_type_input = Variable(torch.from_numpy(self.input_nodes_type_batches[batch])).long()
            edge_input = Variable(torch.from_numpy(self.input_edges_batches[batch])).long()
            answer_tagging_input = Variable(torch.from_numpy(self.input_answer_tagging_batches[batch])).long()
            end_tagging_input = Variable(torch.from_numpy(self.input_end_tagging_batches[batch])).long()
            tgt_input = Variable(torch.from_numpy(self.input_decode_batches[batch])).long()
            tgt_input_length = Variable(torch.from_numpy(self.input_decode_lengths[batch])).long()
            target_valid_node_label = Variable(torch.from_numpy(self.valid_node_label_batches[batch])).long()

            if self.args.cuda:
                src_input = src_input.cuda()
                src_input_length = src_input_length.cuda()
                node_input = node_input.cuda()
                node_type_input = node_type_input.cuda()
                edge_input = edge_input.cuda()
                answer_tagging_input = answer_tagging_input.cuda()
                end_tagging_input = end_tagging_input.cuda()
                tgt_input = tgt_input.cuda()
                tgt_input_length = tgt_input_length.cuda()
                target_valid_node_label = target_valid_node_label.cuda()

            if self.args.beam_size > 1:
                # beam search
                pre_index, selection_logits = model(src_input, src_input_length, answer_tagging_input, end_tagging_input, node_input, node_type_input,
                                        edge_input, tgt_input, tgt_input_length)
            else:
                logits, selection_logits = model(src_input, src_input_length, answer_tagging_input, end_tagging_input, node_input, node_type_input,
                                        edge_input, tgt_input, tgt_input_length)
                pre_index = torch.max(logits, -1)[1]

            for i in range(pre_index.size(0)):
                generate_string = indexs2str(pre_index[i], self.vocabulary_tgt, is_question=True)
                gene_questions.append(generate_string)

            class_index = torch.max(selection_logits, -1)[1]
            node_num = torch.sum(torch.sign(torch.sum(node_input, -1)), -1).cpu()
            target_valid_node_label = target_valid_node_label.cpu().numpy()
            for i in range(class_index.size(0)):
                if self.args.evaluate_complex_data:
                    if ith in self.long_data_indexes:  # long data
                        total_f1_score += f1_score(target_valid_node_label[i][1:node_num[i]],
                                                   class_index[i].cpu()[1:node_num[i]])
                        total_accuracy += self.compute_accuarcy(target_valid_node_label[i][1:node_num[i]],
                                                                class_index[i].cpu()[1:node_num[i]].numpy())
                        total_num += 1
                else:
                    if node_num[i] > 1:    # all data
                        total_f1_score += f1_score(target_valid_node_label[i][1:node_num[i]],
                                                   class_index[i].cpu()[1:node_num[i]])
                        total_accuracy += self.compute_accuarcy(target_valid_node_label[i][1:node_num[i]],
                                                                class_index[i].cpu()[1:node_num[i]].numpy())
                        total_num += 1
                ith += 1

            selected_paths += class_index.cpu().numpy().tolist()


            if batch % 20 == 0:
                print('batch', batch)

        if self.is_test:
            bleu_score, meteor_score, rouge_score, all_bleu_scores, all_meteor_scores, all_rouge_scores = self.compute_BLEU_Meteor_ROUGE(
                gene_questions, self.all_questions, self.all_sentences, long_result=self.args.evaluate_complex_data)
            _, all_spice = self.compute_SPICE(gene_questions, self.spread_spice_ref_questions, long_result=self.args.evaluate_complex_data)
        else:
            bleu_score, meteor_score, rouge_score, all_bleu_scores, all_meteor_scores, all_rouge_scores = self.compute_BLEU_Meteor_ROUGE(gene_questions, self.all_questions, long_result=self.args.evaluate_complex_data)

        f1 = total_f1_score / total_num
        accuracy = total_accuracy / total_num
        print('current validation f1-score & accuracy is', f1, accuracy)

        model.is_test = False
        return bleu_score, meteor_score, rouge_score, accuracy


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

    trainer = Generator_Solver(inited_args, vocabulary_src, vocabulary_tgt)
    evaluator = Test(inited_args, vocabulary_src, vocabulary_tgt, is_test=True)

    trainer.train()
    evaluator.evaluate()

    print(evaluator.args.QG_model_file)
    print('running time', time.time()-start)

