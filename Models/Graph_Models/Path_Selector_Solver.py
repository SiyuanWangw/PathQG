import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score

from Path_Selection_Models.Path_Selector_Model import Selector
import sys
sys.path.append('..')
sys.path.append('../..')
sys.path.append('../../..')
from Models.Graph_Models.Path_Selection_Models.Selection_Config import init_args
from SceneGraph_LoadData import *
import os


class Path_Selector_Solver(object):
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

        model = Selector(self.args, self.word_to_inx_src['PAD'], self.src_embeddings, self.tgt_embeddings)
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Number of training parameters:', total_trainable_params)

        if self.args.cuda:
            model = model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=self.args.lr)
        criterion = nn.CrossEntropyLoss(size_average=False, reduce=False)

        evaluator = Selector_Test(self.args, self.vocabulary_src, self.vocabulary_tgt, is_test=False, is_train=False)
        best_score = 0
        best_epoch = 0

        for epoch in range(self.args.selection_num_epoches):
            # load data
            all_vector_src, all_vector_tgt, \
            input_encode_batches, input_encode_lengths, \
            input_nodes_batches, input_edges_batches, _, \
            input_answer_tagging_batches, input_end_tagging_batches, \
            input_path_tagging_batches, _, valid_node_label_batches, input_decode_batches, \
            _, target_decode_batches, input_decode_lengths, input_nodes_type_batches, \
            input_answer_batches, _, nodes_edges_subpaths_batches = get_path_train_data_each_epoch(params, is_shuffle=True)

            print('epoch:', epoch, ', learning rate:', optimizer.param_groups[0]['lr'])
            n_chunk = len(all_vector_src) // self.args.batch_size

            beta = 1.0
            for batch in range(n_chunk):
                src_input = Variable(torch.from_numpy(input_encode_batches[batch])).long()
                src_input_length = Variable(torch.from_numpy(input_encode_lengths[batch])).long()
                node_input = Variable(torch.from_numpy(input_nodes_batches[batch])).long()
                node_type_input = Variable(torch.from_numpy(input_nodes_type_batches[batch])).long()
                answer_tagging_input = Variable(torch.from_numpy(input_answer_tagging_batches[batch])).long()
                end_tagging_input = Variable(torch.from_numpy(input_end_tagging_batches[batch])).long()
                tgt_input = Variable(torch.from_numpy(input_decode_batches[batch])).long()
                tgt_input_length = Variable(torch.from_numpy(input_decode_lengths[batch])).long()
                target_valid_node_label = Variable(torch.from_numpy(valid_node_label_batches[batch])).long()
                answer_input = Variable(torch.from_numpy(input_answer_batches[batch])).long()

                if self.args.cuda:
                    src_input = src_input.cuda()
                    src_input_length = src_input_length.cuda()
                    node_input = node_input.cuda()
                    node_type_input = node_type_input.cuda()
                    answer_tagging_input = answer_tagging_input.cuda()
                    end_tagging_input = end_tagging_input.cuda()
                    tgt_input = tgt_input.cuda()
                    tgt_input_length = tgt_input_length.cuda()
                    target_valid_node_label = target_valid_node_label.cuda()
                    answer_input = answer_input.cuda()

                logits = model(src_input, src_input_length, answer_tagging_input, end_tagging_input, node_input, node_type_input, tgt_input, tgt_input_length, answer_input)

                select_loss = criterion(logits.view(-1, logits.shape[2]), target_valid_node_label.view(-1)).view(
                    self.args.batch_size, -1)
                mask = torch.sign(torch.sum(node_input, -1)).float()
                select_loss = torch.sum(mask.mul(select_loss)) / torch.sum(mask)

                reg_loss = 0
                for name, param in model.named_parameters():
                    if 'embedding' not in name:
                        reg_loss += self.args.l1_reg * param.abs().sum() + self.args.l2_reg * (param.pow(2)).sum()
                loss = beta*select_loss + reg_loss

                optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm(model.parameters(), self.args.clip_grad)
                optimizer.step()

                if batch % 100 == 0:
                    print(
                        '[INFO] Epoch: %d , batch: %d , selection loss: %.6f, total loss: %.6f, gradient norm: %.6f' % (
                            epoch, batch, select_loss.item(), loss.item(), grad_norm))

            self.adjust_learning_rate(optimizer)
            model.eval()
            f1, accuracy = evaluator.evaluate(model)
            if accuracy > best_score:
                best_score = accuracy
                best_epoch = epoch
                torch.save(model.state_dict(), self.args.Selection_model_file)
                print('model saved!!!!!!!!!')
            print('current best score:', best_score)
            print('current best epoch:', best_epoch)
            model.train()


class Selector_Test(object):
    def __init__(self, inited_args, vocabulary_src, vocabulary_tgt, is_test=False, is_train=False):
        self.args = inited_args
        self.src_embeddings = np.load(self.args.src_embedding_file)
        self.tgt_embeddings = np.load(self.args.tgt_embedding_file)
        self.vocabulary_src = vocabulary_src
        self.vocabulary_tgt = vocabulary_tgt
        self.word_to_inx_src = dict(zip(vocabulary_src, range(len(vocabulary_src))))

        # load data
        if is_train:
            params = get_path_train_data(self.args.batch_size, self.vocabulary_src, self.vocabulary_tgt)
            all_vector_src, all_vector_tgt, \
            self.input_encode_batches, self.input_encode_lengths, \
            self.input_nodes_batches, self.input_edges_batches, _, \
            self.input_answer_tagging_batches, self.input_end_tagging_batches, \
            self.input_path_tagging_batches, _, self.valid_node_label_batches, self.input_decode_batches, \
            _, self.target_decode_batches, self.input_decode_lengths, self.input_nodes_type_batches, \
            self.input_answer_batches, _, self.nodes_edges_subpaths_batches = get_path_train_data_each_epoch(params,is_shuffle=False)
            self.long_data_indexes = np.load(
                os.path.join(dataroot,
                             './processed/SQuAD1.0/Graph_Analysis/SceneGraph/train/long_data_piece_index.npy')).tolist()
            self.file = os.path.join(dataroot, './processed/SQuAD1.0/Graph_Analysis/SceneGraph/train/selected_path_pipe.npy')
        elif is_test is False:
            all_vector_src, all_vector_tgt, \
            self.input_encode_batches, self.input_encode_lengths, \
            self.input_nodes_batches, self.input_edges_batches, _, \
            self.input_answer_tagging_batches, self.input_end_tagging_batches, \
            self.input_path_tagging_batches, _, self.valid_node_label_batches, self.input_decode_batches, \
            self.all_questions, self.input_decode_lengths, self.input_nodes_type_batches, \
            self.input_answer_batches, _, self.nodes_edges_subpaths_batches = get_path_val_data(self.args.batch_size, self.vocabulary_src,
                                                                        self.vocabulary_tgt)
            self.long_data_indexes = np.load(
                os.path.join(dataroot,
                             './processed/SQuAD1.0/Graph_Analysis/SceneGraph/val/long_data_piece_index.npy')).tolist()
            self.file = os.path.join(dataroot, './processed/SQuAD1.0/Graph_Analysis/SceneGraph/val/selected_path_pipe.npy')

        else:
            all_vector_src, all_vector_tgt, \
            self.input_encode_batches, self.input_encode_lengths, \
            self.input_nodes_batches, self.input_edges_batches, _, \
            self.input_answer_tagging_batches, self.input_end_tagging_batches, \
            self.input_path_tagging_batches, _, self.valid_node_label_batches, self.input_decode_batches, \
            self.all_questions, self.input_decode_lengths, self.input_nodes_type_batches, \
            self.input_answer_batches, _, self.all_sentences, self.nodes_edges_subpaths_batches = get_path_test_data(self.args.batch_size,
                                                                                             self.vocabulary_src,
                                                                                             self.vocabulary_tgt)
            self.long_data_indexes = np.load(
                os.path.join(dataroot,
                             './processed/SQuAD1.0/Graph_Analysis/SceneGraph/test/long_data_piece_index.npy')).tolist()
            self.file = os.path.join(dataroot, './processed/SQuAD1.0/Graph_Analysis/SceneGraph/test/selected_path_pipe.npy')

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

    def evaluate(self, model=None):
        if model is None:
            old_model = torch.load(self.args.Selection_model_file)
            model = Selector(self.args, self.word_to_inx_src['PAD'], self.src_embeddings, self.tgt_embeddings)
            model.load_state_dict(old_model)
            model.eval()
        if self.args.cuda:
            model = model.cuda()

        selected_nodes = list()
        total_f1_score = 0
        total_accuracy = 0
        n_chunk = len(self.input_encode_batches)

        ith = 0
        total_num = 0
        for batch in range(n_chunk):
            src_input = Variable(torch.from_numpy(self.input_encode_batches[batch])).long()
            src_input_length = Variable(torch.from_numpy(self.input_encode_lengths[batch])).long()
            node_input = Variable(torch.from_numpy(self.input_nodes_batches[batch])).long()
            node_type_input = Variable(torch.from_numpy(self.input_nodes_type_batches[batch])).long()
            answer_tagging_input = Variable(torch.from_numpy(self.input_answer_tagging_batches[batch])).long()
            end_tagging_input = Variable(torch.from_numpy(self.input_end_tagging_batches[batch])).long()
            tgt_input = Variable(torch.from_numpy(self.input_decode_batches[batch])).long()
            tgt_input_length = Variable(torch.from_numpy(self.input_decode_lengths[batch])).long()
            target_valid_node_label = self.valid_node_label_batches[batch]
            answer_input = Variable(torch.from_numpy(self.input_answer_batches[batch])).long()

            if self.args.cuda:
                src_input = src_input.cuda()
                src_input_length = src_input_length.cuda()
                node_input = node_input.cuda()
                node_type_input = node_type_input.cuda()
                answer_tagging_input = answer_tagging_input.cuda()
                end_tagging_input = end_tagging_input.cuda()
                tgt_input = tgt_input.cuda()
                tgt_input_length = tgt_input_length.cuda()
                answer_input = answer_input.cuda()

            logits = model(src_input, src_input_length, answer_tagging_input, end_tagging_input, node_input, node_type_input, tgt_input,
                           tgt_input_length, answer_input)
            class_index = torch.max(logits, -1)[1]

            node_num = torch.sum(torch.sign(torch.sum(node_input, -1)), -1).cpu()
            for i in range(class_index.size(0)):
                if ith in self.long_data_indexes:
                    total_f1_score += f1_score(target_valid_node_label[i][1:node_num[i]],
                                               class_index[i].cpu()[1:node_num[i]])
                    total_accuracy += self.compute_accuarcy(target_valid_node_label[i][1:node_num[i]],
                                                            class_index[i].cpu()[1:node_num[i]].numpy())
                    total_num += 1
                ith += 1

            selected_nodes += class_index.cpu().numpy().tolist()
            if batch % 100 == 0:
                print('batch', batch)

        f1 = total_f1_score / total_num
        accuracy = total_accuracy / total_num
        print('current validatin f1-score is', f1)
        print('current validation accuracy is', accuracy)

        # np.save(self.file, selected_nodes)

        return f1, accuracy


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

    trainer = Path_Selector_Solver(inited_args, vocabulary_src, vocabulary_tgt)
    evaluator = Selector_Test(inited_args, vocabulary_src, vocabulary_tgt, is_test=True)

    trainer.train()
    evaluator.evaluate()

    print(evaluator.args.Selection_model_file)
    print('running time', time.time()-start)




