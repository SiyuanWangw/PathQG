import pickle
import nltk
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))

import numpy as np
import os
from tqdm import tqdm
import collections

import sys
sys.path.append('../..')
sys.path.append('../../..')
from Data.Preprocess.utils import *

dataroot = os.path.abspath('../../Data')
import time
start = time.time()


def importData(data_type=0):
    if data_type == 0:
        ## training data
        files = (
            './processed/SQuAD1.0/train/sentences.npy', './processed/SQuAD1.0/train/questions.npy',
            './processed/SQuAD1.0/train/answers.npy', './processed/SQuAD1.0/train/answers_start.npy')
    elif data_type == 1:
        ## validation date
        files = (
            './processed/SQuAD1.0/val/sentences.npy', './processed/SQuAD1.0/val/questions.npy',
            './processed/SQuAD1.0/val/answers.npy', './processed/SQuAD1.0/val/answers_start.npy')
    else:
        ## test data
        # files = (
        #     './processed/SQuAD1.0/test/sentences.npy', './processed/SQuAD1.0/test/questions.npy',
        #     './processed/SQuAD1.0/test/answers.npy', './processed/SQuAD1.0/test/answers_start.npy')
        files = (
            './processed/SQuAD1.0/test/rm_sentences.npy', './processed/SQuAD1.0/test/rm_questions.npy',
            './processed/SQuAD1.0/test/rm_answers.npy', './processed/SQuAD1.0/test/rm_answers_start.npy')
    sentences = np.load(os.path.join(dataroot, files[0]))
    questions = np.load(os.path.join(dataroot, files[1]))
    answers = np.load(os.path.join(dataroot, files[2]))
    answers_start = np.load(os.path.join(dataroot, files[3]))
    return sentences, questions, answers, answers_start


def import_graph(data_type=0):
    # load nodes, edges and adjacency from graph
    if data_type == 0:
        graph_file_name = '../../Data/processed/SQuAD1.0/Graph_Analysis/train/enriched_scene_graphs.pkl'
    elif data_type == 1:
        graph_file_name = '../../Data/processed/SQuAD1.0/Graph_Analysis/val/enriched_scene_graphs.pkl'
    else:
        graph_file_name = '../../Data/processed/SQuAD1.0/Graph_Analysis/test/enriched_scene_graphs.pkl'

    fr = open(graph_file_name, 'rb')
    graphs = pickle.load(fr)
    fr.close()

    all_sent_nodes = list()
    all_sent_nodes_type = list()
    all_sent_edges = list()
    all_sent_adjacency = list()
    for each_graph in tqdm(graphs):
        cur_sent_nodes = list()
        cur_sent_nodes_type = list()
        cur_sent_edges = list()
        cur_sent_adjacency = list()
        for j, each_graph_node in enumerate(each_graph.node_list):
            cur_node_tokens = nltk.word_tokenize(each_graph_node.nodetext)
            cur_node_types = each_graph_node.nodetype.lower()
            if(len(nltk.word_tokenize(cur_node_types))) > 1:
                print(cur_node_types)
            cur_sent_nodes.append(cur_node_tokens)
            cur_sent_nodes_type.append(cur_node_types)
            cur_node_edges = list()
            cur_node_adjacency = list()
            for k in range(len(each_graph.node_list)):
                if j == k:
                    cur_node_edges.append(['self-loop'])
                    cur_node_adjacency.append(1)
                elif each_graph.get_edge_by_index(j, k) is not None:
                    cur_edge = each_graph.get_edge_by_index(j, k)
                    cur_node_edges.append(nltk.word_tokenize(cur_edge.edgetext))
                    cur_node_adjacency.append(1)
                elif each_graph.get_edge_by_index(k, j) is not None:
                    cur_edge = each_graph.get_edge_by_index(k, j)
                    cur_node_edges.append(nltk.word_tokenize(cur_edge.edgetext))
                    cur_node_adjacency.append(1)
                else:
                    cur_node_edges.append([])
                    cur_node_adjacency.append(0)
            cur_sent_edges.append(cur_node_edges)
            cur_sent_adjacency.append(cur_node_adjacency)

        cur_sent_adjacency = np.array(cur_sent_adjacency, np.float32)
        if len(cur_sent_adjacency) > 0:
            cur_sent_adjacency = normalize(cur_sent_adjacency)

        all_sent_nodes.append(cur_sent_nodes)
        all_sent_nodes_type.append(cur_sent_nodes_type)
        all_sent_edges.append(cur_sent_edges)
        all_sent_adjacency.append(cur_sent_adjacency)

    if data_type == 0:
        graph_node_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/train/graph_node.npy'
        graph_node_type_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/train/graph_node_type.npy'
        graph_edge_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/train/graph_edge.npy'
        graph_adjacency_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/train/graph_adjacency.npy'
    elif data_type == 1:
        graph_node_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/val/graph_node.npy'
        graph_node_type_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/val/graph_node_type.npy'
        graph_edge_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/val/graph_edge.npy'
        graph_adjacency_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/val/graph_adjacency.npy'
    else:
        graph_node_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/test/graph_node.npy'
        graph_node_type_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/test/graph_node_type.npy'
        graph_edge_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/test/graph_edge.npy'
        graph_adjacency_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/test/graph_adjacency.npy'

    np.save(os.path.join(dataroot, graph_node_file), all_sent_nodes)
    np.save(os.path.join(dataroot, graph_node_type_file), all_sent_nodes_type)
    np.save(os.path.join(dataroot, graph_edge_file), all_sent_edges)
    np.save(os.path.join(dataroot, graph_adjacency_file), all_sent_adjacency)


def normalize(matrix):
    """Row-normalize sparse matrix"""
    # print(matrix)
    rowsum = np.array(matrix.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    matrix = r_mat_inv.dot(matrix)
    # print(matrix)
    return matrix


def import_graph_node_edge(data_type=0):
    if data_type == 0:
        graph_node_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/train/graph_node.npy'
        graph_node_type_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/train/graph_node_type.npy'
        graph_edge_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/train/graph_edge.npy'
        graph_adjacency_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/train/graph_adjacency.npy'
    elif data_type == 1:
        graph_node_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/val/graph_node.npy'
        graph_node_type_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/val/graph_node_type.npy'
        graph_edge_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/val/graph_edge.npy'
        graph_adjacency_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/val/graph_adjacency.npy'
    else:
        graph_node_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/test/graph_node.npy'
        graph_node_type_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/test/graph_node_type.npy'
        graph_edge_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/test/graph_edge.npy'
        graph_adjacency_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/test/graph_adjacency.npy'

    all_sent_nodes = np.load(os.path.join(dataroot, graph_node_file))
    all_sent_nodes_type = np.load(os.path.join(dataroot, graph_node_type_file))
    all_sent_edges = np.load(os.path.join(dataroot, graph_edge_file))
    all_sent_adjacency = np.load(os.path.join(dataroot, graph_adjacency_file))
    return all_sent_nodes, all_sent_nodes_type, all_sent_edges, all_sent_adjacency


def generate_node_edge_vector(all_sent_nodes, all_sent_edges, word_to_inx):
    # generate vectors for nodes and edges
    all_nodes_vector = list()
    all_edges_vector = list()
    for i, cur_sent_nodes in enumerate(all_sent_nodes):
        cur_sent_nodes_vector = list()
        cur_sent_edges_vector = list()

        for j, cur_node_words in enumerate(cur_sent_nodes):
            node_vector = list()
            for each_word in cur_node_words:
                idx = word_to_inx.get(each_word)
                if idx is not None:
                    node_vector.append(idx)
                else:
                    node_vector.append(word_to_inx.get('UNK'))
            cur_sent_nodes_vector.append(node_vector)

            cur_sent_sub_edges_vector = list()
            for k in range(len(cur_sent_nodes)):
                edge_vector = list()
                for each_word_edge in all_sent_edges[i][j][k]:
                    idx = word_to_inx.get(each_word_edge)
                    if idx is not None:
                        edge_vector.append(idx)
                    else:
                        edge_vector.append(word_to_inx.get('UNK'))
                cur_sent_sub_edges_vector.append(edge_vector)
            cur_sent_edges_vector.append(cur_sent_sub_edges_vector)

        all_nodes_vector.append(cur_sent_nodes_vector)
        all_edges_vector.append(cur_sent_edges_vector)
    all_nodes_vector = np.array(all_nodes_vector)
    all_edges_vector = np.array(all_edges_vector)

    return all_nodes_vector, all_edges_vector


def generate_src_vector(src_data_train, words, is_list=False):
    # generate vectors for input texts
    word_to_inx = dict(zip(words, range(len(words))))

    all_vector = []
    for sentence in src_data_train:
        if not is_list:
            words_sentence = nltk.word_tokenize(sentence)
        else:
            words_sentence = sentence
        vector = list()
        for each_word in words_sentence:
            idx = word_to_inx.get(each_word)
            if idx is not None:
                vector.append(idx)
            else:
                vector.append(word_to_inx.get('UNK'))
        all_vector.append(vector)
    all_vector = np.array(all_vector)

    return word_to_inx, all_vector


def generate_tgt_vector(tgt_data_train, words):
    # generate vectors for questions
    word_to_inx = dict(zip(words, range(len(words))))

    all_vector = []
    for question_list in tgt_data_train:
        vector_list = list()
        for question in question_list:
            words_sentence = nltk.word_tokenize(question)
            vector = list()
            for each_word in words_sentence:
                idx = word_to_inx.get(each_word)
                if idx is not None:
                    vector.append(idx)
                else:
                    vector.append(word_to_inx.get('UNK'))
            vector_list.append(vector)
        all_vector.append(vector_list)
    all_vector = np.array(all_vector)

    return word_to_inx, all_vector


def save_import_vectors(is_save=False, is_path=False, data_type=0, is_answer=False):
    # save vectors into files or load vectors from files
    prestr = 'extended_'

    if data_type == 0:
        all_vector_src_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/train/all_vector_src.npy'
        all_nodes_vector_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/train/all_nodes_vector.npy'
        all_nodes_type_vector_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/train/all_nodes_type_vector.npy'
        all_edges_vector_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/train/all_edges_vector.npy'
        all_ques_nodes_vector_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/train/all_ques_nodes_vector.npy'
        all_ques_nodes_type_vector_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/train/all_ques_nodes_type_vector.npy'
        all_ques_edges_vector_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/train/all_ques_edges_vector.npy'
        all_vector_tgt_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/train/all_vector_tgt.npy'
        all_vector_answer_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/train/all_vector_answer.npy'
        node_path_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/train/' + prestr + 'question_node_path.npy'
        edge_path_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/train/' + prestr + 'question_edge_path.npy'
    elif data_type == 1:
        all_vector_src_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/val/all_vector_src.npy'
        all_nodes_vector_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/val/all_nodes_vector.npy'
        all_nodes_type_vector_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/val/all_nodes_type_vector.npy'
        all_edges_vector_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/val/all_edges_vector.npy'
        all_ques_nodes_vector_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/val/all_ques_nodes_vector.npy'
        all_ques_nodes_type_vector_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/val/all_ques_nodes_type_vector.npy'
        all_ques_edges_vector_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/val/all_ques_edges_vector.npy'
        all_vector_tgt_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/val/all_vector_tgt.npy'
        all_vector_answer_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/val/all_vector_answer.npy'
        node_path_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/val/' + prestr + 'question_node_path.npy'
        edge_path_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/val/' + prestr + 'question_edge_path.npy'
    else:
        all_vector_src_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/test/all_vector_src.npy'
        all_nodes_vector_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/test/all_nodes_vector.npy'
        all_nodes_type_vector_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/test/all_nodes_type_vector.npy'
        all_edges_vector_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/test/all_edges_vector.npy'
        all_ques_nodes_vector_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/test/all_ques_nodes_vector.npy'
        all_ques_nodes_type_vector_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/test/all_ques_nodes_type_vector.npy'
        all_ques_edges_vector_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/test/all_ques_edges_vector.npy'
        all_vector_tgt_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/test/all_vector_tgt.npy'
        all_vector_answer_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/test/all_vector_answer.npy'
        node_path_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/test/' + prestr + 'question_node_path.npy'
        edge_path_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/test/' + prestr + 'question_edge_path.npy'

    if is_save:
        sentences, questions, answers, _ = importData(data_type=data_type)
        # re_sentences = replace_answer(sentences, answers) # for answer separation model
        all_sent_nodes, all_sent_nodes_type, all_sent_edges, _ = import_graph_node_edge(data_type=data_type)

        graph_src_vocabulary = np.load(
            os.path.join(dataroot, './processed/SQuAD1.0/Graph_Analysis/SceneGraph/graph_vocabulary_src.npy'))
        graph_tgt_vocabulary = np.load(
            os.path.join(dataroot, './processed/SQuAD1.0/Graph_Analysis/SceneGraph/graph_vocabulary_tgt.npy'))

        word_to_inx_src, all_vector_src = generate_src_vector(sentences, graph_src_vocabulary)
        # word_to_inx_src, all_vector_src = generate_tgt_vector(re_sentences, graph_src_vocabulary) # for answer separation model
        print('src vector generated!')
        _, all_vector_tgt = generate_tgt_vector(questions, graph_tgt_vocabulary)
        print('tgt vector generated!')
        _, all_vector_answer = generate_tgt_vector(answers, graph_src_vocabulary)
        print('answer vector generated!')
        np.save(os.path.join(dataroot, all_vector_src_file), all_vector_src)
        np.save(os.path.join(dataroot, all_vector_tgt_file), all_vector_tgt)
        np.save(os.path.join(dataroot, all_vector_answer_file), all_vector_answer)

        if is_path:
            all_sent_ques_nodes = np.load(os.path.join(dataroot, node_path_file))
            all_sent_ques_edges = np.load(os.path.join(dataroot, edge_path_file))

            all_ques_nodes_vector, all_ques_nodes_type_vector, all_ques_edges_vector = generate_ques_node_edge_vector(all_sent_ques_nodes, all_sent_ques_edges, word_to_inx_src)
            print('path vector generated!')
            np.save(os.path.join(dataroot, all_ques_nodes_vector_file), all_ques_nodes_vector)
            np.save(os.path.join(dataroot, all_ques_nodes_type_vector_file), all_ques_nodes_type_vector)
            np.save(os.path.join(dataroot, all_ques_edges_vector_file), all_ques_edges_vector)
        else:
            all_nodes_vector, all_edges_vector = generate_node_edge_vector(all_sent_nodes, all_sent_edges,
                                                                       word_to_inx_src)
            _, all_nodes_type_vector = generate_src_vector(all_sent_nodes_type, graph_src_vocabulary, is_list=True)
            print('graph vector generated!')
            np.save(os.path.join(dataroot, all_nodes_vector_file), all_nodes_vector)
            np.save(os.path.join(dataroot, all_nodes_type_vector_file), all_nodes_type_vector)
            np.save(os.path.join(dataroot, all_edges_vector_file), all_edges_vector)

    else:
        all_vector_src = np.load(os.path.join(dataroot, all_vector_src_file))
        all_vector_tgt = np.load(os.path.join(dataroot, all_vector_tgt_file))
        if is_path:
            all_nodes_vector = np.load(os.path.join(dataroot, all_ques_nodes_vector_file))
            all_nodes_type_vector = np.load(os.path.join(dataroot, all_ques_nodes_type_vector_file))
            all_edges_vector = np.load(os.path.join(dataroot, all_ques_edges_vector_file))

            if not is_answer:
                return all_vector_src, all_nodes_vector, all_nodes_type_vector, all_edges_vector, all_vector_tgt
            else:
                all_vector_answer = np.load(os.path.join(dataroot, all_vector_answer_file))
                return all_vector_src, all_vector_answer, all_nodes_vector, all_nodes_type_vector, all_edges_vector, all_vector_tgt

        else:
            all_nodes_vector = np.load(os.path.join(dataroot, all_nodes_vector_file))
            all_nodes_type_vector = np.load(os.path.join(dataroot, all_nodes_type_vector_file))
            all_edges_vector = np.load(os.path.join(dataroot, all_edges_vector_file))

            if not is_answer:
                return all_vector_src, all_nodes_vector, all_nodes_type_vector, all_edges_vector, all_vector_tgt
            else:
                all_vector_answer = np.load(os.path.join(dataroot, all_vector_answer_file))
                return all_vector_src, all_vector_answer, all_nodes_vector, all_nodes_type_vector, all_edges_vector, all_vector_tgt


def find_end_path(data_type=0):
    # find the end entity and path between answer and end of each sentence
    sentences, questions, answers, answers_start = importData(data_type=data_type)
    end_list = list()
    node_path_list = list ()
    edge_path_list = list ()
    path_valid_rate = list()
    path_valid_length = list()

    if data_type == 0:
        graph_file_name = '../../Data/processed/SQuAD1.0/Graph_Analysis/train/enriched_scene_graphs.pkl'
        end_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/train/question_end.npy'
        node_path_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/train/question_node_path.npy'
        edge_path_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/train/question_edge_path.npy'
    elif data_type == 1:
        graph_file_name = '../../Data/processed/SQuAD1.0/Graph_Analysis/val/enriched_scene_graphs.pkl'
        end_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/val/question_end.npy'
        node_path_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/val/question_node_path.npy'
        edge_path_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/val/question_edge_path.npy'
    else:
        graph_file_name = '../../Data/processed/SQuAD1.0/Graph_Analysis/test/enriched_scene_graphs.pkl'
        end_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/test/question_end.npy'
        node_path_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/test/question_node_path.npy'
        edge_path_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/test/question_edge_path.npy'

    fr = open(graph_file_name, 'rb')
    graphs = pickle.load(fr)
    fr.close()

    long_num = 0
    for i in tqdm(range(len(questions))):
        cur_sent_node_num = len(graphs[i].node_text_list)
        cur_sent_end_list = list()
        cur_sent_node_path_list = list()
        cur_sent_edge_path_list = list()
        for j in range(len(questions[i])):
            cur_answer_start_index = len(nltk.word_tokenize(sentences[i][:answers_start[i][j]]))
            cur_answer_end_index = cur_answer_start_index + len(nltk.word_tokenize(answers[i][j]))
            cur_ques_end_candidates = list()
            cur_ques_node_path = list()
            cur_ques_edge_path = list()
            valid_path_node_rate = 0

            answer_neigbors = list()
            true_answer_neighbors = list()
            for k in range(cur_sent_node_num):
                cur_node_span_bounds = graphs[i].node_list[k].nodespan_bounds
                if (cur_node_span_bounds[0] >= cur_answer_start_index and cur_node_span_bounds[1] <= cur_answer_end_index) or \
                    (cur_node_span_bounds[0] <= cur_answer_start_index and cur_node_span_bounds[1] > cur_answer_start_index) or \
                    (cur_node_span_bounds[0] < cur_answer_end_index and cur_node_span_bounds[1] >= cur_answer_end_index):
                    answer_neigbors.append(graphs[i].node_list[k])
                    true_answer_neighbors.append(graphs[i].node_list[k])
            if len(answer_neigbors) > 0:
                m = 1
                help_answer_neighbors = answer_neigbors
                while len(answer_neigbors) < cur_sent_node_num and len(help_answer_neighbors) > 0:
                    help_answer_neighbors = list()
                    for each_node in answer_neigbors:
                        for each_neighbor_node in (each_node.preNodes + each_node.succNodes):
                            if each_neighbor_node not in answer_neigbors and each_neighbor_node not in help_answer_neighbors:
                                help_answer_neighbors.append(each_neighbor_node)
                                each_neighbor_node.add_parent(each_node)
                                if each_neighbor_node.nodetext in questions[i][j]:
                                    cur_ques_end_candidates.append((each_neighbor_node, m))
                    answer_neigbors = answer_neigbors + help_answer_neighbors
                    m += 1

                if len(cur_ques_end_candidates) > 0:
                    end_candidate = cur_ques_end_candidates[-1][0]
                    cur_sent_end_list.append(end_candidate)
                    for n in range(cur_ques_end_candidates[-1][1]):
                        if end_candidate in end_candidate.parent.preNodes:
                            cur_ques_node_path.append(end_candidate)
                            cur_ques_edge_path.append(graphs[i].get_edge(end_candidate, end_candidate.parent))
                        elif end_candidate in end_candidate.parent.succNodes:
                            cur_ques_node_path.append(end_candidate)
                            cur_ques_edge_path.append(graphs[i].get_edge(end_candidate.parent, end_candidate))
                        else:
                            raise Exception('wrong!!!')
                        if n > 0:
                            if end_candidate.nodetext in questions[i][j]:
                                valid_path_node_rate += 1
                        end_candidate = end_candidate.parent
                    if end_candidate not in true_answer_neighbors:
                        print('Wrong!!')

                    cur_ques_node_path.append(end_candidate)
                    cur_ques_node_path.reverse()
                    cur_ques_edge_path.reverse()

                    if cur_ques_node_path[0].nodespan_bounds[0] >= cur_ques_node_path[-1].nodespan_bounds[0]:
                        cur_ques_node_path.reverse()
                        cur_ques_edge_path.reverse()

                    cur_sent_node_path_list.append(cur_ques_node_path)
                    cur_sent_edge_path_list.append(cur_ques_edge_path)

                    if len(cur_ques_node_path) > 3: # answer+end
                        long_num += 1
                        path_valid_rate.append(valid_path_node_rate)
                        path_valid_length.append(len(cur_ques_node_path)-2)
                else:
                    cur_sent_end_list.append(None)
                    cur_sent_node_path_list.append([])
                    cur_sent_edge_path_list.append([])
            else:
                cur_sent_end_list.append(None)
                cur_sent_node_path_list.append([])
                cur_sent_edge_path_list.append([])

        end_list.append(cur_sent_end_list)
        node_path_list.append(cur_sent_node_path_list)
        edge_path_list.append(cur_sent_edge_path_list)

    print(long_num)
    print(np.sum(path_valid_rate)/np.sum(path_valid_length), np.mean(path_valid_rate), np.mean(path_valid_length))
    np.save(os.path.join(dataroot, end_file), end_list)
    np.save(os.path.join(dataroot, node_path_file), node_path_list)
    np.save(os.path.join(dataroot, edge_path_file), edge_path_list)


def extend_node_edge_to_path(data_type=0):
    # extend the path with missing entity
    if data_type == 0:
        graph_file_name = '../../Data/processed/SQuAD1.0/Graph_Analysis/train/enriched_scene_graphs.pkl'
        node_path_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/train/question_node_path.npy'
        edge_path_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/train/question_edge_path.npy'
        extended_node_path_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/train/extended_question_node_path.npy'
        extended_edge_path_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/train/extended_question_edge_path.npy'
    elif data_type == 1:
        graph_file_name = '../../Data/processed/SQuAD1.0/Graph_Analysis/val/enriched_scene_graphs.pkl'
        node_path_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/val/question_node_path.npy'
        edge_path_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/val/question_edge_path.npy'
        extended_node_path_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/val/extended_question_node_path.npy'
        extended_edge_path_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/val/extended_question_edge_path.npy'
    else:
        graph_file_name = '../../Data/processed/SQuAD1.0/Graph_Analysis/test/enriched_scene_graphs.pkl'
        node_path_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/test/question_node_path.npy'
        edge_path_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/test/question_edge_path.npy'
        extended_node_path_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/test/extended_question_node_path.npy'
        extended_edge_path_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/test/extended_question_edge_path.npy'

    fr = open(graph_file_name, 'rb')
    graphs = pickle.load(fr)
    fr.close()

    sentences, _, _, _ = importData(data_type=data_type)

    node_path_list = np.load(os.path.join(dataroot, node_path_file))
    edge_path_list = np.load(os.path.join(dataroot, edge_path_file))
    extended_node_path_list = list()
    extended_edge_path_list = list()

    for i in tqdm(range(len(node_path_list))):
        cur_sent_extended_node_path_list = list()
        cur_sent_extended_edge_path_list = list()

        for j in range(len(node_path_list[i])):
            cur_node_path = node_path_list[i][j]
            cur_edge_path = edge_path_list[i][j]
            cur_extended_ques_node_path = [each for each in cur_node_path]
            cur_extended_ques_node_path_spanstart = [each.nodespan_bounds[0] for each in cur_node_path]
            cur_extended_ques_edge_path = [each for each in cur_edge_path]

            if len(cur_node_path) > 1:
                if cur_node_path[0].nodespan_bounds[0] < cur_node_path[-1].nodespan_bounds[0]:

                    start_index = cur_node_path[0].nodespan_bounds[1]
                    end_index = cur_node_path[-1].nodespan_bounds[0]
                    for k in range(len(cur_node_path)):
                        each_node = cur_node_path[k]
                        for each_neighbor_node in (each_node.preNodes + each_node.succNodes):
                            if each_neighbor_node.nodespan_bounds[0] > start_index and \
                                    each_neighbor_node.nodespan_bounds[1] <= end_index:
                                if each_neighbor_node not in cur_extended_ques_node_path:
                                    insert_index = get_insert_index(cur_extended_ques_node_path_spanstart, each_neighbor_node.nodespan_bounds)
                                    cur_extended_ques_node_path.insert(insert_index, each_neighbor_node)
                                    cur_extended_ques_node_path_spanstart.insert(insert_index, each_neighbor_node.nodespan_bounds[0])
                                    if each_neighbor_node in each_node.preNodes:
                                        cur_extended_ques_edge_path.insert(insert_index, graphs[i].get_edge(each_neighbor_node, each_node))
                                    elif each_neighbor_node in each_node.succNodes:
                                        cur_extended_ques_edge_path.insert(insert_index-1, graphs[i].get_edge(each_node, each_neighbor_node))

                else:
                    start_index = cur_node_path[-1].nodespan_bounds[1]
                    end_index = cur_node_path[0].nodespan_bounds[0]
                    for k in range(len(cur_node_path)-1, -1 -1):
                        each_node = cur_node_path[k]
                        for each_neighbor_node in (each_node.preNodes + each_node.succNodes):
                            if each_neighbor_node.nodespan_bounds[0] > start_index and \
                                    each_neighbor_node.nodespan_bounds[1] <= end_index:
                                if each_neighbor_node not in cur_extended_ques_node_path:
                                    insert_index = get_insert_index(cur_extended_ques_node_path_spanstart, each_neighbor_node.nodespan_bounds, is_reverse=True)
                                    cur_extended_ques_node_path.insert(insert_index, each_neighbor_node)
                                    cur_extended_ques_node_path_spanstart.insert(insert_index, each_neighbor_node.nodespan_bounds[0])
                                    if each_neighbor_node in each_node.preNodes:
                                        cur_extended_ques_edge_path.insert(insert_index, graphs[i].get_edge(each_neighbor_node, each_node))
                                    elif each_neighbor_node in each_node.succNodes:
                                        cur_extended_ques_edge_path.insert(insert_index, graphs[i].get_edge(each_node, each_neighbor_node))
            cur_sent_extended_node_path_list.append(cur_extended_ques_node_path)
            cur_sent_extended_edge_path_list.append(cur_extended_ques_edge_path)
        extended_node_path_list.append(cur_sent_extended_node_path_list)
        extended_edge_path_list.append(cur_sent_extended_edge_path_list)

    np.save(os.path.join(dataroot, extended_node_path_file), extended_node_path_list)
    np.save(os.path.join(dataroot, extended_edge_path_file), extended_edge_path_list)


def get_insert_index(li, elem_spanbounds, is_reverse=False):
    if not is_reverse:
        for i in range(len(li)-1):
            if elem_spanbounds[0] > li[i] and elem_spanbounds[1] <= li[i+1]:
                return i+1
    else:
        for i in range(len(li)-1, 0, -1):
            if elem_spanbounds[0] > li[i] and elem_spanbounds[1] <= li[i-1]:
                return i


def end_path_tagging(data_type=0):
    # tag the sentence with given end entity and path (node+edge)
    sentences, questions, _, _ = importData(data_type=data_type)
    prestr = ''
    if data_type == 0:
        end_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/train/' + prestr + 'question_end.npy'
        node_path_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/train/' + prestr + 'question_node_path.npy'
        edge_path_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/train/' + prestr + 'question_edge_path.npy'
        end_tagging_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/train/' + prestr + 'question_end_tagging.npy'
        path_tagging_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/train/' + prestr + 'question_path_tagging.npy'
    elif data_type == 1:
        end_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/val/' + prestr + 'question_end.npy'
        node_path_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/val/' + prestr + 'question_node_path.npy'
        edge_path_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/val/' + prestr + 'question_edge_path.npy'
        end_tagging_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/val/' + prestr + 'question_end_tagging.npy'
        path_tagging_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/val/' + prestr + 'question_path_tagging.npy'
    else:
        end_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/test/' + prestr + 'question_end.npy'
        node_path_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/test/' + prestr + 'question_node_path.npy'
        edge_path_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/test/' + prestr + 'question_edge_path.npy'
        end_tagging_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/test/' + prestr + 'question_end_tagging.npy'
        path_tagging_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/test/' + prestr + 'question_path_tagging.npy'

    end_list = np.load(os.path.join(dataroot, end_file))
    node_path_list = np.load(os.path.join(dataroot, node_path_file))
    edge_path_list = np.load(os.path.join(dataroot, edge_path_file))
    end_tagging_list = list()
    path_tagging_list = list()

    for i in tqdm(range(len(sentences))):
        cur_sent_tokens = nltk.word_tokenize(sentences[i])
        for j in range(len(questions[i])):
            cur_ques_end_tagging = [0,]*len(cur_sent_tokens)
            cur_ques_path_tagging = [0,]*len(cur_sent_tokens)
            if end_list[i][j] is not None:
                end_text = end_list[i][j].nodetext
                if end_text in sentences[i]:
                    c_index = sentences[i].find(end_text)
                    w_index = len(nltk.word_tokenize(sentences[i][:c_index]))
                    cur_ques_end_tagging[w_index: w_index + len(nltk.word_tokenize(end_text))] = [1, ] * len(nltk.word_tokenize(end_text))
                else:
                    end_text_token = nltk.word_tokenize(end_text)
                    for each_token in end_text_token:
                        if each_token in cur_sent_tokens:
                            cur_ques_end_tagging[cur_sent_tokens.index(each_token)] = 1

                for each_path_node in node_path_list[i][j]:
                    each_path_node_text = each_path_node.nodetext
                    each_path_node_tokens = nltk.word_tokenize(each_path_node_text)
                    if each_path_node_text in sentences[i]:
                        c_index = sentences[i].find(each_path_node_text)
                        w_index = len(nltk.word_tokenize(sentences[i][:c_index]))
                        cur_ques_path_tagging[w_index: w_index + len(each_path_node_tokens)] = [1, ] * len(each_path_node_tokens)
                    else:
                        for each_token in each_path_node_tokens:
                            if each_token in cur_sent_tokens:
                                cur_ques_path_tagging[cur_sent_tokens.index(each_token)] = 1

                for each_path_edge in edge_path_list[i][j]:
                    each_path_edge_text = each_path_edge.edgetext
                    each_path_edge_tokens = nltk.word_tokenize(each_path_edge_text)
                    if each_path_edge_text in sentences[i]:
                        c_index = sentences[i].find(each_path_edge_text)
                        w_index = len(nltk.word_tokenize(sentences[i][:c_index]))
                        cur_ques_path_tagging[w_index: w_index + len(each_path_edge_tokens)] = [2, ] * len(each_path_edge_tokens)
                    else:
                        for each_token in each_path_edge_tokens:
                            if each_token in cur_sent_tokens:
                                cur_ques_path_tagging[cur_sent_tokens.index(each_token)] = 2

            end_tagging_list.append(cur_ques_end_tagging)
            path_tagging_list.append(cur_ques_path_tagging)

    np.save(os.path.join(dataroot, end_tagging_file), end_tagging_list)
    np.save(os.path.join(dataroot, path_tagging_file), path_tagging_list)


def get_selected_path_tagging(data_type=0):
    # tag the sentence with ground truth path components
    sentences, questions, answers, _ = importData(data_type=data_type)
    prestr = 'extended_'

    if data_type == 0:
        selected_node_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/train/question_valid_node_label.npy'  # ground truth path
        selected_edge_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/train/question_valid_edge_label.npy'
        node_path_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/train/' + prestr + 'question_node_path.npy'
        edge_path_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/train/' + prestr + 'question_edge_path.npy'
        selected_path_tagging_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/train/question_selected_path_tagging.npy'
    elif data_type == 1:
        selected_node_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/val/question_valid_node_label.npy'
        selected_edge_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/val/question_valid_edge_label.npy'
        node_path_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/val/' + prestr + 'question_node_path.npy'
        edge_path_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/val/' + prestr + 'question_edge_path.npy'
        selected_path_tagging_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/val/question_selected_path_tagging.npy'
    else:
        selected_node_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/test/question_valid_node_label.npy'
        selected_edge_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/test/question_valid_edge_label.npy'
        node_path_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/test/' + prestr + 'question_node_path.npy'
        edge_path_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/test/' + prestr + 'question_edge_path.npy'
        selected_path_tagging_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/test/question_selected_path_tagging.npy'

    selected_node_label = np.load(os.path.join(dataroot, selected_node_file))
    selected_edge_label = np.load(os.path.join(dataroot, selected_edge_file))
    node_path_list = np.load(os.path.join(dataroot, node_path_file))
    edge_path_list = np.load(os.path.join(dataroot, edge_path_file))
    path_tagging_list = list()

    ith = 0
    for i in tqdm(range(len(sentences))):
        cur_sent_tokens = nltk.word_tokenize(sentences[i])
        for j in range(len(questions[i])):
            cur_ques_path_tagging = [0,]*len(cur_sent_tokens)

            if ith < len(selected_node_label):
                for k, label in enumerate(selected_node_label[ith]):
                    if k < len(node_path_list[i][j]) and label == 1:
                        each_path_node = node_path_list[i][j][k]
                        each_path_node_text = each_path_node.nodetext
                        each_path_node_tokens = nltk.word_tokenize(each_path_node_text)

                        if each_path_node_text in sentences[i]:
                            c_index = sentences[i].find(each_path_node_text)
                            w_index = len(nltk.word_tokenize(sentences[i][:c_index]))
                            cur_ques_path_tagging[w_index: w_index + len(each_path_node_tokens)] = [1, ] * len(
                                each_path_node_tokens)

            if ith < len(selected_edge_label):
                for k, label in enumerate(selected_edge_label[ith]):
                    if k < len(edge_path_list[i][j]) and label == 1:
                        each_path_edge_text = edge_path_list[i][j][k].edgetext
                        each_path_edge_tokens = nltk.word_tokenize(each_path_edge_text)

                        if each_path_edge_text in sentences[i]:
                            c_index = sentences[i].find(each_path_edge_text)
                            w_index = len(nltk.word_tokenize(sentences[i][:c_index]))
                            cur_ques_path_tagging[w_index: w_index + len(each_path_edge_tokens)] = [1, ] * len(each_path_edge_tokens)

            path_tagging_list.append(cur_ques_path_tagging)

    np.save(os.path.join(dataroot, selected_path_tagging_file), path_tagging_list)


def get_selected_path_tagging_ours(data_type=0):
    # tag the sentence with our selected path components
    sentences, questions, answers, _ = importData(data_type=data_type)
    prestr = 'extended_'

    if data_type == 0:
        selected_node_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/train/selected_path_pipe.npy'
        node_path_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/train/' + prestr + 'question_node_path.npy'
        edge_path_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/train/' + prestr + 'question_edge_path.npy'
        selected_path_tagging_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/train/our_selected_path_tagging.npy'
    elif data_type == 1:
        selected_node_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/val/selected_path_pipe.npy'
        node_path_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/val/' + prestr + 'question_node_path.npy'
        edge_path_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/val/' + prestr + 'question_edge_path.npy'
        selected_path_tagging_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/val/our_selected_path_tagging.npy'
    else:
        selected_node_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/test/selected_path_pipe.npy'
        node_path_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/test/' + prestr + 'question_node_path.npy'
        edge_path_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/test/' + prestr + 'question_edge_path.npy'
        selected_path_tagging_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/test/our_selected_path_tagging.npy'

    selected_node_label = np.load(os.path.join(dataroot, selected_node_file))
    node_path_list = np.load(os.path.join(dataroot, node_path_file))
    edge_path_list = np.load(os.path.join(dataroot, edge_path_file))
    path_tagging_list = list()

    ith = 0
    for i in tqdm(range(len(sentences))):
        cur_sent_tokens = nltk.word_tokenize(sentences[i])
        for j in range(len(questions[i])):
            cur_ques_path_tagging = [0,]*len(cur_sent_tokens)

            cur_path_nodes_edges = list()
            for k in range(len(node_path_list[i][j])):
                cur_path_nodes_edges.append(node_path_list[i][j][k].nodetext)
                if k < len(edge_path_list[i][j]):
                    cur_path_nodes_edges.append(edge_path_list[i][j][k].edgetext)

            if ith < len(selected_node_label):
                for m, label in enumerate(selected_node_label[ith]):
                    if m < len(cur_path_nodes_edges):
                        if label == 1:
                            each_path_node_text = cur_path_nodes_edges[m]
                            each_path_node_tokens = nltk.word_tokenize(each_path_node_text)

                            if each_path_node_text in sentences[i]:
                                c_index = sentences[i].find(each_path_node_text)
                                w_index = len(nltk.word_tokenize(sentences[i][:c_index]))
                                cur_ques_path_tagging[w_index: w_index + len(each_path_node_tokens)] = [1, ] * len(
                                    each_path_node_tokens)
                            else:
                                for each_token in each_path_node_tokens:
                                    if each_token in cur_sent_tokens:
                                        cur_ques_path_tagging[cur_sent_tokens.index(each_token)] = 1

            path_tagging_list.append(cur_ques_path_tagging)
            ith += 1
    print(ith)

    np.save(os.path.join(dataroot, selected_path_tagging_file), path_tagging_list)


def generate_ques_node_edge_vector(all_sent_ques_nodes, all_sent_ques_edges, word_to_inx):
    # generate vectors for question path

    all_nodes_vector = list()
    all_nodes_type_vector = list()
    all_edges_vector = list()
    for i in tqdm(range(len(all_sent_ques_nodes))):
        cur_sent_ques_nodes_vector = list()
        cur_sent_ques_nodes_type_vector = list()
        cur_sent_ques_edges_vector = list()

        for j in range(len(all_sent_ques_nodes[i])):
            cur_ques_nodes_vector = list()
            cur_ques_nodes_type_vector = list()
            cur_ques_edges_vector = list()

            for k, cur_node in enumerate(all_sent_ques_nodes[i][j]):
                cur_node_words = nltk.word_tokenize(cur_node.nodetext)
                node_vector = list()
                for each_word in cur_node_words:
                    idx = word_to_inx.get(each_word)
                    if idx is not None:
                        node_vector.append(idx)
                    else:
                        node_vector.append(word_to_inx.get('UNK'))
                cur_ques_nodes_vector.append(node_vector)

                if k < len(all_sent_ques_edges[i][j]):
                    cur_edge_words = nltk.word_tokenize(all_sent_ques_edges[i][j][k].edgetext)
                    edge_vector = list()
                    for each_word_edge in cur_edge_words:
                        idx = word_to_inx.get(each_word_edge)
                        if idx is not None:
                            edge_vector.append(idx)
                        else:
                            edge_vector.append(word_to_inx.get('UNK'))
                    cur_ques_nodes_vector.append(edge_vector)

                cur_node_type = cur_node.nodetype.lower()
                type_idx = word_to_inx.get(cur_node_type)
                if type_idx is not None:
                    cur_ques_nodes_type_vector.append(type_idx)
                else:
                    cur_ques_nodes_type_vector.append(word_to_inx.get('UNK'))

            for cur_edge in all_sent_ques_edges[i][j]:
                cur_edge_words = nltk.word_tokenize(cur_edge.edgetext)
                edge_vector = list()
                for each_word_edge in cur_edge_words:
                    idx = word_to_inx.get(each_word_edge)
                    if idx is not None:
                        edge_vector.append(idx)
                    else:
                        edge_vector.append(word_to_inx.get('UNK'))
                cur_ques_edges_vector.append(edge_vector)

            cur_sent_ques_nodes_vector.append(cur_ques_nodes_vector)
            cur_sent_ques_nodes_type_vector.append(cur_ques_nodes_type_vector)
            cur_sent_ques_edges_vector.append(cur_ques_edges_vector)

        all_nodes_vector.append(cur_sent_ques_nodes_vector)
        all_nodes_type_vector.append(cur_sent_ques_nodes_type_vector)
        all_edges_vector.append(cur_sent_ques_edges_vector)

    all_nodes_vector = np.array(all_nodes_vector)
    all_nodes_type_vector = np.array(all_nodes_type_vector)
    all_edges_vector = np.array(all_edges_vector)

    return all_nodes_vector, all_nodes_type_vector, all_edges_vector


def get_valid_node_labels(data_type=0):
    _, questions, _, _ = importData(data_type=data_type)
    prestr = 'extended_'

    if data_type == 0:
        node_path_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/train/' + prestr + 'question_node_path.npy'
        edge_path_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/train/' + prestr + 'question_edge_path.npy'
        valid_node_label_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/train/question_valid_node_label.npy'
        valid_edge_label_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/train/question_valid_edge_label.npy'
        node_answer_end_tag_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/train/question_node_answer_end_tagging.npy'
    elif data_type == 1:
        node_path_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/val/' + prestr + 'question_node_path.npy'
        edge_path_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/val/' + prestr + 'question_edge_path.npy'
        valid_node_label_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/val/question_valid_node_label.npy'
        valid_edge_label_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/val/question_valid_edge_label.npy'
        node_answer_end_tag_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/val/question_node_answer_end_tagging.npy'
    else:
        node_path_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/test/' + prestr + 'question_node_path.npy'
        edge_path_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/test/' + prestr + 'question_edge_path.npy'
        valid_node_label_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/test/question_valid_node_label.npy'
        valid_edge_label_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/test/question_valid_edge_label.npy'
        node_answer_end_tag_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/test/question_node_answer_end_tagging.npy'

    all_sent_ques_nodes = np.load(os.path.join(dataroot, node_path_file))
    all_sent_ques_edges = np.load(os.path.join(dataroot, edge_path_file))

    all_valid_node_labels = list()
    all_valid_edge_labels = list()
    all_node_answer_end_taggings = list()
    valid_num = 0
    for i in tqdm(range(len(all_sent_ques_nodes))):
        for j in range(len(all_sent_ques_nodes[i])):
            cur_ques_valid_node_labels = list()
            cur_ques_valid_edge_labels = list()
            cur_ques_node_answer_end_taggings = list()

            for n, cur_node in enumerate(all_sent_ques_nodes[i][j]):
                if n > 0 and n < (len(all_sent_ques_nodes[i][j])-1):
                    cur_node_tokens = nltk.word_tokenize(cur_node.nodetext)
                    if cur_node.nodetext in questions[i][j]:
                        cur_ques_valid_node_labels.append(1)
                        valid_num += 1
                    else:
                        overlap = False
                        for each_token in cur_node_tokens:
                            if each_token not in stopwords and each_token in questions[i][j]:
                                overlap = True
                                break
                        if overlap is True:
                            cur_ques_valid_node_labels.append(1)
                            valid_num += 1
                        else:
                            cur_ques_valid_node_labels.append(0)
                    cur_ques_node_answer_end_taggings.append(0)
                else:
                    cur_ques_valid_node_labels.append(1)
                    if n == 0:
                        cur_ques_node_answer_end_taggings.append(1)
                    else:
                        cur_ques_node_answer_end_taggings.append(2)

            for m, cur_edge in enumerate(all_sent_ques_edges[i][j]):
                cur_edge_tokens = nltk.word_tokenize(cur_edge.edgetext)
                if cur_edge.edgetext in questions[i][j]:
                    cur_ques_valid_edge_labels.append(1)
                    valid_num += 1
                else:
                    overlap = False
                    for each_token in cur_edge_tokens:
                        if each_token not in stopwords and each_token in questions[i][j]:
                            overlap = True
                            break
                    if overlap is True:
                        cur_ques_valid_edge_labels.append(1)
                        valid_num += 1
                    else:
                        cur_ques_valid_edge_labels.append(0)

            all_valid_edge_labels.append(cur_ques_valid_edge_labels)
            all_node_answer_end_taggings.append(cur_ques_node_answer_end_taggings)

            cur_ques_valid_path_labels = list()
            for k in range(len(cur_ques_valid_node_labels)):
                cur_ques_valid_path_labels.append(cur_ques_valid_node_labels[k])
                if k < len(cur_ques_valid_edge_labels):
                    cur_ques_valid_path_labels.append(cur_ques_valid_edge_labels[k])
            all_valid_node_labels.append(cur_ques_valid_path_labels)

    all_valid_node_labels = np.array(all_valid_node_labels)
    all_valid_edge_labels = np.array(all_valid_edge_labels)
    all_node_answer_end_taggings = np.array(all_node_answer_end_taggings)
    np.save(os.path.join(dataroot, valid_node_label_file), all_valid_node_labels)
    np.save(os.path.join(dataroot, valid_edge_label_file), all_valid_edge_labels)
    np.save(os.path.join(dataroot, node_answer_end_tag_file), all_node_answer_end_taggings)


def split_long_short_path_length_data(data_type=0):

    prestr = 'extended_'

    if data_type == 0:
        node_path_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/train/' + prestr + 'question_node_path.npy'
        long_short_data_tag_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/train/long_data_tag.npy'
        long_data_index_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/train/long_data_piece_index.npy'
    elif data_type == 1:
        node_path_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/val/' + prestr + 'question_node_path.npy'
        long_short_data_tag_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/val/long_data_tag.npy'
        long_data_index_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/val/long_data_piece_index.npy'
    else:
        node_path_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/test/' + prestr + 'question_node_path.npy'
        long_short_data_tag_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/test/long_data_tag.npy'
        long_data_index_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/test/long_data_piece_index.npy'

    all_sent_ques_nodes = np.load(os.path.join(dataroot, node_path_file))
    long_data_index = list()
    long_short_data_tag = list()

    more_than_4_num = 0
    ith = 0
    for i in range(len(all_sent_ques_nodes)):
        cur_sent_tag = list()
        for j in range(len(all_sent_ques_nodes[i])):
            if len(all_sent_ques_nodes[i][j]) >= 4:
                long_data_index.append(ith)
                cur_sent_tag.append(1)
                more_than_4_num += 1
            else:
                cur_sent_tag.append(0)
            ith += 1

        long_short_data_tag.append(cur_sent_tag)
    print(more_than_4_num, ith, more_than_4_num/ith)

    long_data_index = np.array(long_data_index)
    long_short_data_tag = np.array(long_short_data_tag)
    np.save(os.path.join(dataroot, long_data_index_file), long_data_index)
    np.save(os.path.join(dataroot, long_short_data_tag_file), long_short_data_tag)


def get_long_data(data_type=0, is_save=False):
    prestr = 'extended_'

    if data_type == 0:
        long_text_files = (
            './processed/SQuAD1.0/Graph_Analysis/SceneGraph/long/train/sentences.npy', './processed/SQuAD1.0/Graph_Analysis/SceneGraph/long/train/questions.npy',
            './processed/SQuAD1.0/Graph_Analysis/SceneGraph/long/train/answers.npy', './processed/SQuAD1.0/Graph_Analysis/SceneGraph/long/train/answers_start.npy',
            './processed/SQuAD1.0/Graph_Analysis/SceneGraph/long/train/nodes.npy', './processed/SQuAD1.0/Graph_Analysis/SceneGraph/long/train/edges.npy',
            './processed/SQuAD1.0/Graph_Analysis/SceneGraph/long/train/adjacency.npy', './processed/SQuAD1.0/Graph_Analysis/SceneGraph/long/train/nodes_type.npy')
        graph_file = ('./processed/SQuAD1.0/Graph_Analysis/SceneGraph/train/' + prestr + 'question_node_path.npy', './processed/SQuAD1.0/Graph_Analysis/SceneGraph/train/' + prestr + 'question_edge_path.npy')
        long_graph_file = ('./processed/SQuAD1.0/Graph_Analysis/SceneGraph/long/train/' + prestr + 'question_node_path.npy', './processed/SQuAD1.0/Graph_Analysis/SceneGraph/long/train/' + prestr + 'question_edge_path.npy')
        long_data_tag_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/train/long_data_tag.npy'
    elif data_type == 1:
        long_text_files = (
            './processed/SQuAD1.0/Graph_Analysis/SceneGraph/long/val/sentences.npy', './processed/SQuAD1.0/Graph_Analysis/SceneGraph/long/val/questions.npy',
            './processed/SQuAD1.0/Graph_Analysis/SceneGraph/long/val/answers.npy', './processed/SQuAD1.0/Graph_Analysis/SceneGraph/long/val/answers_start.npy',
            './processed/SQuAD1.0/Graph_Analysis/SceneGraph/long/val/nodes.npy', './processed/SQuAD1.0/Graph_Analysis/SceneGraph/long/val/edges.npy',
            './processed/SQuAD1.0/Graph_Analysis/SceneGraph/long/val/adjacency.npy', './processed/SQuAD1.0/Graph_Analysis/SceneGraph/long/val/nodes_type.npy')
        graph_file = ('./processed/SQuAD1.0/Graph_Analysis/SceneGraph/val/' + prestr + 'question_node_path.npy', './processed/SQuAD1.0/Graph_Analysis/SceneGraph/val/' + prestr + 'question_edge_path.npy')
        long_graph_file = ('./processed/SQuAD1.0/Graph_Analysis/SceneGraph/long/val/' + prestr + 'question_node_path.npy', './processed/SQuAD1.0/Graph_Analysis/SceneGraph/long/val/' + prestr + 'question_edge_path.npy')
        long_data_tag_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/val/long_data_tag.npy'
    else:
        long_text_files = (
            './processed/SQuAD1.0/Graph_Analysis/SceneGraph/long/test/sentences.npy', './processed/SQuAD1.0/Graph_Analysis/SceneGraph/long/test/questions.npy',
            './processed/SQuAD1.0/Graph_Analysis/SceneGraph/long/test/answers.npy', './processed/SQuAD1.0/Graph_Analysis/SceneGraph/long/test/answers_start.npy',
            './processed/SQuAD1.0/Graph_Analysis/SceneGraph/long/test/nodes.npy', './processed/SQuAD1.0/Graph_Analysis/SceneGraph/long/test/edges.npy',
            './processed/SQuAD1.0/Graph_Analysis/SceneGraph/long/test/adjacency.npy', './processed/SQuAD1.0/Graph_Analysis/SceneGraph/long/test/nodes_type.npy')
        graph_file = ('./processed/SQuAD1.0/Graph_Analysis/SceneGraph/test/' + prestr + 'question_node_path.npy', './processed/SQuAD1.0/Graph_Analysis/SceneGraph/test/' + prestr + 'question_edge_path.npy')
        long_graph_file = ('./processed/SQuAD1.0/Graph_Analysis/SceneGraph/long/test/' + prestr + 'question_node_path.npy', './processed/SQuAD1.0/Graph_Analysis/SceneGraph/long/test/' + prestr + 'question_edge_path.npy')
        long_data_tag_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/test/long_data_tag.npy'

    if is_save:
        sentences, questions, answers, answers_start = importData(data_type=data_type)
        all_sent_nodes, all_sent_nodes_type, all_sent_edges, all_sent_adjacency = import_graph_node_edge(data_type=data_type)
        long_data_tag = np.load(os.path.join(dataroot, long_data_tag_file))
        question_node_path = np.load(os.path.join(dataroot, graph_file[0]))
        question_edge_path = np.load(os.path.join(dataroot, graph_file[1]))

        long_sentences, long_questions, long_answers, long_answers_start, long_all_nodes, long_all_nodes_type, long_all_edges, long_all_adjacency = list(), list(), list(), list(), list(), list(), list(), list()
        long_node_paths, long_edge_paths = list(), list()

        for i in range(len(sentences)):
            cur_sent_long_ques = list()
            cur_sent_long_answer = list()
            cur_sent_long_answer_start = list()
            cur_sent_long_node_path = list()
            cur_sent_long_edge_path = list()

            for j in range(len(questions[i])):
                if long_data_tag[i][j] == 1:
                    cur_sent_long_ques.append(questions[i][j])
                    cur_sent_long_answer.append(answers[i][j])
                    cur_sent_long_answer_start.append(answers_start[i][j])
                    cur_sent_long_node_path.append(question_node_path[i][j])
                    cur_sent_long_edge_path.append(question_edge_path[i][j])

            if len(cur_sent_long_ques) > 0:
                long_sentences.append(sentences[i])
                long_questions.append(cur_sent_long_ques)
                long_answers.append(cur_sent_long_answer)
                long_answers_start.append(cur_sent_long_answer_start)
                long_all_nodes.append(all_sent_nodes[i])
                long_all_nodes_type.append(all_sent_nodes_type[i])
                long_all_edges.append(all_sent_edges[i])
                long_all_adjacency.append(all_sent_adjacency[i])
                long_node_paths.append(cur_sent_long_node_path)
                long_edge_paths.append(cur_sent_long_edge_path)

        print(long_node_paths[0])
        print(long_edge_paths[0])
        print('*'*100)

        print(sum([len(each) for each in long_questions]))
        np.save(os.path.join(dataroot, long_text_files[0]), np.array(long_sentences))
        np.save(os.path.join(dataroot, long_text_files[1]), np.array(long_questions))
        np.save(os.path.join(dataroot, long_text_files[2]), np.array(long_answers))
        np.save(os.path.join(dataroot, long_text_files[3]), np.array(long_answers_start))
        np.save(os.path.join(dataroot, long_text_files[4]), np.array(long_all_nodes))
        np.save(os.path.join(dataroot, long_text_files[7]), np.array(long_all_nodes_type))
        np.save(os.path.join(dataroot, long_text_files[5]), np.array(long_all_edges))
        np.save(os.path.join(dataroot, long_text_files[6]), np.array(long_all_adjacency))
        np.save(os.path.join(dataroot, long_graph_file[0]), np.array(long_node_paths))
        np.save(os.path.join(dataroot, long_graph_file[1]), np.array(long_edge_paths))

    else:
        long_sentences = np.load(os.path.join(dataroot, long_text_files[0]))
        long_questions = np.load(os.path.join(dataroot, long_text_files[1]))
        long_answers = np.load(os.path.join(dataroot, long_text_files[2]))
        long_answers_start = np.load(os.path.join(dataroot, long_text_files[3]))
        long_all_nodes = np.load(os.path.join(dataroot, long_text_files[4]))
        long_all_edges = np.load(os.path.join(dataroot, long_text_files[5]))
        long_all_adjacency = np.load(os.path.join(dataroot, long_text_files[6]))
        long_all_nodes_type = np.load(os.path.join(dataroot, long_text_files[7]))

        return long_sentences, long_questions, long_answers, long_answers_start, long_all_nodes, long_all_edges, long_all_adjacency, long_all_nodes_type


def spread_vector(all_vector_src, all_nodes_vector, all_edges_vector, all_adjacency, all_vector_tgt, questions, is_path=False, all_nodes_type_vector=None, answer_vector=None, sentences=None):
    # spread the all vector for each sentence
    spread_all_vector_src = list()
    spread_all_nodes_vector = list()
    spread_all_nodes_type_vector = list()
    spread_all_edges_vector = list()
    apread_all_adjacency = list()
    spread_all_vector_tgt = list()
    spread_all_vector_answer = list()
    spread_questions = list()
    spread_sentences = list()

    for i, each_src_tgts in enumerate(all_vector_tgt):
        for j, each_tgt in enumerate(each_src_tgts):
            spread_all_vector_src.append(all_vector_src[i])
            # spread_all_vector_src.append(all_vector_src[i][j])

            if not is_path:
                spread_all_nodes_vector.append(all_nodes_vector[i])
                spread_all_edges_vector.append(all_edges_vector[i])
                if all_nodes_type_vector is not None:
                    spread_all_nodes_type_vector.append(all_nodes_type_vector[i])
            else:
                spread_all_nodes_vector.append(all_nodes_vector[i][j])
                spread_all_edges_vector.append(all_edges_vector[i][j])
                if all_nodes_type_vector is not None:
                    spread_all_nodes_type_vector.append(all_nodes_type_vector[i][j])
            apread_all_adjacency.append(all_adjacency[i])
            spread_all_vector_tgt.append(each_tgt)
            if answer_vector is not None:
                spread_all_vector_answer.append(answer_vector[i][j])
            spread_questions.append(questions[i])
            if sentences is not None:
                spread_sentences.append(sentences[i])

    if sentences is None:
        if all_nodes_type_vector is None and answer_vector is None:
            return np.array(spread_all_vector_src), np.array(spread_all_nodes_vector), np.array(spread_all_edges_vector), np.array(apread_all_adjacency), \
                   np.array(spread_all_vector_tgt), np.array(spread_questions)
        elif all_nodes_type_vector is not None and answer_vector is None:
            return np.array(spread_all_vector_src), np.array(spread_all_nodes_vector), np.array(spread_all_edges_vector), np.array(apread_all_adjacency), \
                   np.array(spread_all_vector_tgt), np.array(spread_questions), np.array(spread_all_nodes_type_vector)
        elif all_nodes_type_vector is None and answer_vector is not None:
            return np.array(spread_all_vector_src), np.array(spread_all_nodes_vector), np.array(spread_all_edges_vector), np.array(apread_all_adjacency), \
                   np.array(spread_all_vector_tgt), np.array(spread_questions), np.array(spread_all_vector_answer)
        else:
            return np.array(spread_all_vector_src), np.array(spread_all_nodes_vector), np.array(spread_all_edges_vector), np.array(apread_all_adjacency), \
                   np.array(spread_all_vector_tgt), np.array(spread_questions), np.array(spread_all_nodes_type_vector), np.array(spread_all_vector_answer)
    else:
        if all_nodes_type_vector is None and answer_vector is None:
            return np.array(spread_all_vector_src), np.array(spread_all_nodes_vector), np.array(spread_all_edges_vector), np.array(apread_all_adjacency), \
                   np.array(spread_all_vector_tgt), np.array(spread_questions), np.array(spread_sentences)
        elif all_nodes_type_vector is not None and answer_vector is None:
            return np.array(spread_all_vector_src), np.array(spread_all_nodes_vector), np.array(spread_all_edges_vector), np.array(apread_all_adjacency), \
                   np.array(spread_all_vector_tgt), np.array(spread_questions), np.array(spread_all_nodes_type_vector), np.array(spread_sentences)
        elif all_nodes_type_vector is None and answer_vector is not None:
            return np.array(spread_all_vector_src), np.array(spread_all_nodes_vector), np.array(spread_all_edges_vector), np.array(apread_all_adjacency), \
                   np.array(spread_all_vector_tgt), np.array(spread_questions), np.array(spread_all_vector_answer), np.array(spread_sentences)
        else:
            return np.array(spread_all_vector_src), np.array(spread_all_nodes_vector), np.array(spread_all_edges_vector), np.array(apread_all_adjacency), \
                   np.array(spread_all_vector_tgt), np.array(spread_questions), np.array(spread_all_nodes_type_vector), np.array(spread_all_vector_answer), np.array(spread_sentences)


def get_graph_batches(batch_size, all_vector_src, all_nodes_vector, all_edges_vector, all_adjacency,
                      all_vector_tgt, answer_taggings, graph_tagging, distance_tagging,
                      neighbor_taggings, valid_node_label, word_to_inx_src, word_to_inx_tgt,
                      index=None, all_nodes_type_vector=None, all_node_answer_end_tagging=None,
                      is_path=True, all_vector_answer=None):
    if index is not None:
        all_vector_src = all_vector_src[index]
        all_nodes_vector = all_nodes_vector[index]
        if all_nodes_type_vector is not None:
            all_nodes_type_vector = all_nodes_type_vector[index]
        all_edges_vector = all_edges_vector[index]
        all_adjacency = all_adjacency[index]
        all_vector_tgt = all_vector_tgt[index]
        answer_taggings = answer_taggings[index]
        graph_tagging = graph_tagging[index]
        distance_tagging = distance_tagging[index]
        neighbor_taggings = neighbor_taggings[index]
        valid_node_label = valid_node_label[index]
        if all_node_answer_end_tagging is not None:
            all_node_answer_end_tagging = all_node_answer_end_tagging[index]
        if all_vector_answer is not None:
            all_vector_answer = all_vector_answer[index]

    n_chunk = len(all_vector_src) // batch_size
    src_x_batches = []
    src_x_lengths = []
    nodes_batches = []
    nodes_type_batches = []
    edges_batches = []
    adjacency_batches = []
    answer_tagging_batches = []
    graph_tagging_batches = []
    distance_tagging_batches = []
    neighbor_tagging_batches = []
    valid_node_label_batches = []
    node_answer_end_tagging_batches = []
    answer_batches = []

    tgt_x_batches = []
    tgt_y_batches = []
    tgt_x_lengths = []
    answer_lengthes = []

    for i in range(n_chunk):
        start_index = i * batch_size
        end_index = start_index + batch_size

        src_batch = all_vector_src[start_index:end_index]
        node_batch = all_nodes_vector[start_index:end_index]
        if all_nodes_type_vector is not None:
            node_type_batch = all_nodes_type_vector[start_index:end_index]
        edge_batch = all_edges_vector[start_index:end_index]
        adj_batch = all_adjacency[start_index: end_index]
        tgt_batch = all_vector_tgt[start_index:end_index]
        answer_tagging_batch = answer_taggings[start_index: end_index]
        graph_tagging_batch = graph_tagging[start_index: end_index]
        distance_tagging_batch = distance_tagging[start_index: end_index]
        neighbor_tagging_batch = neighbor_taggings[start_index: end_index]
        valid_node_label_batch = valid_node_label[start_index: end_index]
        if all_node_answer_end_tagging is not None:
            node_answer_end_tagging_batch = all_node_answer_end_tagging[start_index:end_index]
        if all_vector_answer is not None:
            answer_batch = all_vector_answer[start_index:end_index]

        src_length = max(map(len, src_batch))
        if src_length > 100:
            src_length = 100
        tgt_length = 30 #20
        answer_length = 5

        # node_num = max(map(len, node_batch))
        edge_num = max(map(len, edge_batch))
        node_num = edge_num + 1
        if node_num > 20:
            node_num = 20

        all_nodes_length = list()
        for each_sent_nodes in node_batch:
            for each_node in each_sent_nodes:
                all_nodes_length.append(len(each_node))
        if len(all_nodes_length) > 0:
            node_length = max(all_nodes_length)
        else:
            node_length = 1
        # 8
        if node_length > 6:
            node_length = 6

        all_edges_length = list()
        for each_sent_edges in edge_batch:
            for each_edge in each_sent_edges:
                all_edges_length.append(len(each_edge))
        if len(all_edges_length) > 0:
            edge_length = max(all_edges_length)
        else:
            edge_length = 1
        # 10
        if edge_length > 8:
            edge_length = 8

        src_length_list = list(map(len, src_batch))
        src_length_list = [src_length if each > src_length else each for each in src_length_list]
        tgt_length_list = list(map(len, tgt_batch))
        tgt_length_list = [tgt_length if each > tgt_length else each for each in tgt_length_list]
        if all_vector_answer is not None:
            answer_length_list = list(map(len, answer_batch))
            answer_length_list = [answer_length if each > answer_length else each for each in answer_length_list]
        else:
            answer_length_list = []

        src_x_data = np.full((batch_size, src_length), word_to_inx_src['PAD'], np.int32)
        answer_tagging_data = np.full((batch_size, src_length), 0, np.int32)
        graph_tagging_data = np.full((batch_size, src_length), 0, np.int32)
        distance_tagging_data = np.full((batch_size, src_length), 0, np.int32)
        neighbor_tagging_data = np.full((batch_size, node_num), 0.0, np.float32)
        valid_node_label_data = np.full((batch_size, node_num*2-1), 0, np.int32)
        node_answer_end_tagging_data = np.full((batch_size, node_num), 0, np.int32)
        node_type_data = np.full((batch_size, node_num), word_to_inx_src['PAD'], np.int32)
        answer_data = np.full((batch_size, answer_length), word_to_inx_src['PAD'], np.int32)

        # print(src_batch[0])
        for row in range(batch_size):
            if len(src_batch[row]) <= src_length:
                src_x_data[row, :len(src_batch[row])] = src_batch[row]
            else:
                src_x_data[row] = src_batch[row][:src_length]

            if len(answer_tagging_batch[row]) <= src_length:
                answer_tagging_data[row, :len(answer_tagging_batch[row])] = answer_tagging_batch[row]
            else:
                answer_tagging_data[row] = answer_tagging_batch[row][:src_length]

            if len(graph_tagging_batch[row]) <= src_length:
                graph_tagging_data[row, :len(graph_tagging_batch[row])] = graph_tagging_batch[row]
            else:
                graph_tagging_data[row] = graph_tagging_batch[row][:src_length]

            if len(distance_tagging_batch[row]) <= src_length:
                distance_tagging_data[row, :len(distance_tagging_batch[row])] = distance_tagging_batch[row]
            else:
                distance_tagging_data[row] = distance_tagging_batch[row][:src_length]

            if len(neighbor_tagging_batch[row]) <= node_num:
                neighbor_tagging_data[row, :len(neighbor_tagging_batch[row])] = neighbor_tagging_batch[row]
            else:
                neighbor_tagging_data[row] = neighbor_tagging_batch[row][:node_num]

            # if len(valid_node_label_batch[row]) <= node_num:
            #     valid_node_label_data[row, :len(valid_node_label_batch[row])] = valid_node_label_batch[row]
            # else:
            #     valid_node_label_data[row] = valid_node_label_batch[row][:node_num]
            if len(valid_node_label_batch[row]) <= node_num*2-1:
                valid_node_label_data[row, :len(valid_node_label_batch[row])] = valid_node_label_batch[row]
            else:
                valid_node_label_data[row] = valid_node_label_batch[row][:node_num*2-1]

            if all_nodes_type_vector is not None:
                if len(node_type_batch[row]) <= node_num:
                    node_type_data[row, :len(node_type_batch[row])] = node_type_batch[row]
                else:
                    node_type_data[row] = node_type_batch[row][:node_num]

            if all_node_answer_end_tagging is not None:
                if len(node_answer_end_tagging_batch[row]) <= node_num:
                    node_answer_end_tagging_data[row, : len(node_answer_end_tagging_batch[row])] = node_answer_end_tagging_batch[row]
                else:
                    node_answer_end_tagging_data[row] = node_answer_end_tagging_batch[row][:node_num]

            if all_vector_answer is not None:
                if len(answer_batch[row]) <= answer_length:
                    answer_data[row, :len(answer_batch[row])] = answer_batch[row]
                else:
                    answer_data[row] = answer_batch[row][:answer_length]

        # node_data = np.full((batch_size, node_num, node_length), word_to_inx_src['PAD'], np.int32)
        node_data = np.full((batch_size, 2*node_num-1, node_length), word_to_inx_src['PAD'], np.int32)
        ###
        edge_data = np.full((batch_size, node_num-1, edge_length), word_to_inx_src['PAD'], np.int32)
        # edge_data = np.full((batch_size, node_num, node_num, edge_length), word_to_inx_src['PAD'], np.int32)

        adj_data = np.full((batch_size, node_num, node_num), 0.0, np.float32)
        for row in range(batch_size):
            for col in range(node_num*2-1):
                if col < len(node_batch[row]):
                    if len(node_batch[row][col]) <= node_length:
                        node_data[row, col, :len(node_batch[row][col])] = node_batch[row][col]
                    else:
                        node_data[row, col] = node_batch[row][col][:node_length]

            for col in range(node_num):
                if is_path:
                    if col < (node_num-1):
                        if col < len(edge_batch[row]):
                            if len(edge_batch[row][col]) <= edge_length:
                                edge_data[row, col, :len(edge_batch[row][col])] = edge_batch[row][col]
                            else:
                                edge_data[row, col] = edge_batch[row][col][:edge_length]

        tgt_x_data = np.full((batch_size, tgt_length), word_to_inx_tgt['PAD'], np.int32)
        for row in range(batch_size):
            if len(tgt_batch[row]) <= tgt_length:
                tgt_x_data[row, :len(tgt_batch[row])] = tgt_batch[row]
            else:
                tgt_x_data[row] = tgt_batch[row][:tgt_length]

        tgt_y_data = np.copy(tgt_x_data)
        tgt_y_data[:, :-1] = tgt_x_data[:, 1:]

        src_x_batches.append(src_x_data)
        src_x_lengths.append(np.array(src_length_list))
        nodes_batches.append(node_data)
        nodes_type_batches.append(node_type_data)
        ###
        edges_batches.append(edge_data)

        adjacency_batches.append(adj_data)
        answer_tagging_batches.append(answer_tagging_data)
        graph_tagging_batches.append(graph_tagging_data)
        distance_tagging_batches.append(distance_tagging_data)
        neighbor_tagging_batches.append(neighbor_tagging_data)
        valid_node_label_batches.append(valid_node_label_data)
        node_answer_end_tagging_batches.append(node_answer_end_tagging_data)

        tgt_x_batches.append(tgt_x_data)
        tgt_y_batches.append(tgt_y_data)
        tgt_x_lengths.append(np.array(tgt_length_list))

        answer_batches.append(answer_data)
        answer_lengthes.append(np.array(answer_length_list))

    if all_vector_answer is None:
        return src_x_batches, src_x_lengths, nodes_batches, edges_batches, adjacency_batches, answer_tagging_batches, graph_tagging_batches, \
               distance_tagging_batches, neighbor_tagging_batches, valid_node_label_batches, tgt_x_batches, tgt_y_batches, tgt_x_lengths, \
               nodes_type_batches, node_answer_end_tagging_batches
    else:
        return src_x_batches, src_x_lengths, nodes_batches, edges_batches, adjacency_batches, answer_tagging_batches, \
               graph_tagging_batches, distance_tagging_batches, neighbor_tagging_batches, valid_node_label_batches, \
               tgt_x_batches, tgt_y_batches, tgt_x_lengths, nodes_type_batches, node_answer_end_tagging_batches, answer_batches, answer_lengthes


def get_graph_train_data(batch_size, vocabulary_src, vocabulary_tgt):
    train_sentences, train_questions, train_answers, _ = importData(data_type=0)
    train_all_sent_nodes, _, train_all_sent_edges, train_all_sent_adjacency = import_graph_node_edge(data_type=0)

    word_to_inx_src = dict(zip(vocabulary_src, range(len(vocabulary_src))))
    word_to_inx_tgt = dict(zip(vocabulary_tgt, range(len(vocabulary_tgt))))

    all_vector_src, all_vector_answer, all_nodes_vector, _, all_edges_vector, all_vector_tgt = save_import_vectors(is_save=False, data_type=0, is_answer=True)
    train_answer_taggings = np.load(os.path.join(dataroot, './processed/SQuAD1.0/train/answer_labels.npy'))
    all_answer_taggings = np.array(train_answer_taggings)
    all_neighbor_tagging = all_answer_taggings
    train_graph_tagging = np.load(os.path.join(dataroot, './processed/SQuAD1.0/Graph_Analysis/SceneGraph/train/question_end_tagging.npy'))
    all_graph_tagging = np.array(train_graph_tagging)

    train_distance_tagging = np.load(os.path.join(dataroot, './processed/SQuAD1.0/Graph_Analysis/SceneGraph/train/question_selected_path_tagging.npy'))  # Upper bound model
    all_distance_tagging = np.array(train_distance_tagging)

    print(len(all_answer_taggings), len(all_graph_tagging), len(all_distance_tagging))
    print(len(all_vector_src), len(all_nodes_vector), len(all_edges_vector), len(all_vector_tgt))

    # spread all questions
    all_vector_src, all_nodes_vector, all_edges_vector, all_adjacency, all_vector_tgt, _, all_vector_answer, _ = spread_vector(all_vector_src, all_nodes_vector, all_edges_vector, train_all_sent_adjacency, all_vector_tgt, train_questions, answer_vector=all_vector_answer, sentences=train_sentences)
    print(len(all_vector_src), len(all_nodes_vector), len(all_edges_vector), len(all_vector_tgt), len(all_vector_answer))

    params = (batch_size, all_vector_src, all_nodes_vector, all_edges_vector, \
              all_adjacency, all_vector_tgt, all_answer_taggings, \
              all_graph_tagging, all_distance_tagging, all_neighbor_tagging, \
              all_neighbor_tagging, word_to_inx_src, word_to_inx_tgt, all_vector_answer)

    return params


def get_graph_train_data_each_epoch(params, is_shuffle=True):
    batch_size, all_vector_src, all_nodes_vector, all_edges_vector, \
    all_adjacency, all_vector_tgt, all_answer_taggings, \
    all_graph_tagging, all_distance_tagging, all_neighbor_tagging, \
    all_neighbor_tagging, word_to_inx_src, word_to_inx_tgt, all_vector_answer = params

    index = np.arange(len(all_vector_src))
    if is_shuffle:
        print('Shuffle')
        np.random.shuffle(index)

    input_encode_batches, input_encode_lengths, input_nodes_batches, \
    input_edges_batches, input_adj_batches, input_answer_tagging_batches, \
    input_graph_tagging_batches, input_distance_tagging_batches, \
    input_neighbor_tagging_batches, _, input_decode_batches, \
    target_decode_batches, input_decode_lengths, _, _, input_answer_batches, input_answer_lengthes \
        = get_graph_batches(batch_size, all_vector_src, all_nodes_vector,
                            all_edges_vector, all_adjacency, all_vector_tgt,
                            all_answer_taggings,  all_graph_tagging,
                            all_distance_tagging, all_neighbor_tagging, all_neighbor_tagging,
                            word_to_inx_src, word_to_inx_tgt, index=index, is_path=False, all_vector_answer=all_vector_answer)

    return all_vector_src, all_vector_tgt, input_encode_batches, input_encode_lengths, \
           input_nodes_batches, input_edges_batches, input_adj_batches, \
           input_answer_tagging_batches, input_graph_tagging_batches, \
           input_distance_tagging_batches, input_neighbor_tagging_batches, \
           input_decode_batches, None, target_decode_batches, input_answer_batches, input_answer_lengthes


def get_graph_val_data(batch_size, vocabulary_src, vocabulary_tgt):
    val_sentences, val_questions, val_answers, _ = importData(data_type=1)
    val_all_sent_nodes, _, val_all_sent_edges, val_all_sent_adjacency = import_graph_node_edge(data_type=1)

    word_to_inx_src = dict(zip(vocabulary_src, range(len(vocabulary_src))))
    word_to_inx_tgt = dict(zip(vocabulary_tgt, range(len(vocabulary_tgt))))

    all_vector_src, all_vector_answer, all_nodes_vector, _, all_edges_vector, all_vector_tgt = save_import_vectors(is_save=False, data_type=1, is_answer=True)
    val_answer_taggings = np.load(os.path.join(dataroot, './processed/SQuAD1.0/val/answer_labels.npy'))
    all_answer_taggings = np.array(val_answer_taggings)
    all_neighbor_tagging = all_answer_taggings
    val_graph_tagging = np.load(os.path.join(dataroot, './processed/SQuAD1.0/Graph_Analysis/SceneGraph/val/question_end_tagging.npy'))
    all_graph_tagging = np.array(val_graph_tagging)

    val_distance_tagging = np.load(os.path.join(dataroot, './processed/SQuAD1.0/Graph_Analysis/SceneGraph/val/question_selected_path_tagging.npy'))
    all_distance_tagging = np.array(val_distance_tagging)

    all_vector_src, all_nodes_vector, all_edges_vector, all_adjacency, all_vector_tgt, all_questions, all_vector_answer, _ = spread_vector(all_vector_src, all_nodes_vector, all_edges_vector, val_all_sent_adjacency, all_vector_tgt, val_questions, answer_vector=all_vector_answer, sentences=val_sentences)
    print(len(all_vector_src), len(all_nodes_vector), len(all_edges_vector), len(all_vector_tgt), len(all_vector_answer))

    input_encode_batches, input_encode_lengths, input_nodes_batches, \
    input_edges_batches, input_adj_batches, input_answer_tagging_batches, \
    input_graph_tagging_batches, input_distance_tagging_batches, \
    input_neighbor_tagging_batches, _, input_decode_batches, _, \
    input_decode_lengths, _, _, input_answer_batches, input_answer_lengthes\
        = get_graph_batches(batch_size, all_vector_src, all_nodes_vector,
                            all_edges_vector, all_adjacency, all_vector_tgt,
                            all_answer_taggings, all_graph_tagging, all_distance_tagging,
                            all_neighbor_tagging, all_neighbor_tagging, word_to_inx_src,
                            word_to_inx_tgt, is_path=False, all_vector_answer=all_vector_answer)

    return all_vector_src, all_vector_tgt, input_encode_batches, input_encode_lengths, \
           input_nodes_batches, input_edges_batches, input_adj_batches, \
           input_answer_tagging_batches, input_graph_tagging_batches, \
           input_distance_tagging_batches, input_neighbor_tagging_batches, \
           input_decode_batches, all_questions, input_answer_batches, input_answer_lengthes


def get_graph_test_data(batch_size, vocabulary_src, vocabulary_tgt):
    test_sentences, test_questions, test_answers, _ = importData(data_type=2)
    test_all_sent_nodes, _, test_all_sent_edges, test_all_sent_adjacency = import_graph_node_edge(data_type=2)

    word_to_inx_src = dict(zip(vocabulary_src, range(len(vocabulary_src))))
    word_to_inx_tgt = dict(zip(vocabulary_tgt, range(len(vocabulary_tgt))))

    all_vector_src, all_vector_answer, all_nodes_vector, _, all_edges_vector, all_vector_tgt = save_import_vectors(is_save=False, data_type=2,  is_answer=True)
    test_answer_taggings = np.load(os.path.join(dataroot, './processed/SQuAD1.0/test/answer_labels.npy'))
    all_answer_taggings = np.array(test_answer_taggings)
    all_neighbor_tagging = all_answer_taggings
    test_graph_tagging = np.load(os.path.join(dataroot, './processed/SQuAD1.0/Graph_Analysis/SceneGraph/test/question_end_tagging.npy'))
    all_graph_tagging = np.array(test_graph_tagging)

    test_distance_tagging = np.load(os.path.join(dataroot, './processed/SQuAD1.0/Graph_Analysis/SceneGraph/test/question_selected_path_tagging.npy'))
    all_distance_tagging = np.array(test_distance_tagging)

    all_vector_src, all_nodes_vector, all_edges_vector, all_adjacency, all_vector_tgt, all_questions, all_vector_answer, all_sentences = spread_vector(all_vector_src, all_nodes_vector, all_edges_vector, test_all_sent_adjacency, all_vector_tgt, test_questions, answer_vector=all_vector_answer, sentences=test_sentences)
    print(len(all_vector_src), len(all_nodes_vector), len(all_edges_vector), len(all_vector_tgt), len(all_vector_answer))
    spread_spice_ref_questions = list()
    for i in range(len(test_questions)):
        for j in range(len(test_questions[i])):
            spread_spice_ref_questions.append(test_questions[i][j])

    input_encode_batches, input_encode_lengths, input_nodes_batches, \
    input_edges_batches, input_adj_batches, input_answer_tagging_batches, \
    input_graph_tagging_batches, input_distance_tagging_batches, \
    input_neighbor_tagging_batches, _, input_decode_batches, _, \
    input_decode_lengths, _, _, input_answer_batches, input_answer_lengthes \
        = get_graph_batches(batch_size, all_vector_src, all_nodes_vector,
                            all_edges_vector, all_adjacency, all_vector_tgt,
                            all_answer_taggings, all_graph_tagging,
                            all_distance_tagging, all_neighbor_tagging, all_neighbor_tagging,
                            word_to_inx_src, word_to_inx_tgt, is_path=False, all_vector_answer=all_vector_answer)

    return all_vector_src, all_vector_tgt, input_encode_batches, input_encode_lengths, \
           input_nodes_batches, input_edges_batches, input_adj_batches, \
           input_answer_tagging_batches, input_graph_tagging_batches, \
           input_distance_tagging_batches, input_neighbor_tagging_batches, \
           input_decode_batches, all_questions, input_answer_batches, input_answer_lengthes, \
           all_sentences, spread_spice_ref_questions


def get_path_train_data(batch_size, vocabulary_src, vocabulary_tgt):
    train_sentences, train_questions, train_answers, _ = importData(data_type=0)
    _, train_all_sent_nodes_type, _, train_all_sent_adjacency = import_graph_node_edge(data_type=0)

    word_to_inx_src = dict(zip(vocabulary_src, range(len(vocabulary_src))))
    word_to_inx_tgt = dict(zip(vocabulary_tgt, range(len(vocabulary_tgt))))

    all_vector_src, all_vector_answer, all_nodes_vector, all_nodes_type_vector, all_edges_vector, all_vector_tgt = save_import_vectors(is_save=False, is_path=True, data_type=0, is_answer=True)
    train_answer_taggings = np.load(os.path.join(dataroot, './processed/SQuAD1.0/train/answer_labels.npy'))
    all_answer_taggings = np.array(train_answer_taggings)
    all_neighbor_tagging = all_answer_taggings

    train_end_tagging = np.load(os.path.join(dataroot, './processed/SQuAD1.0/Graph_Analysis/SceneGraph/train/question_end_tagging.npy'))
    all_end_tagging = np.array(train_end_tagging)
    all_path_tagging = all_end_tagging
    train_valid_node_label = np.load(os.path.join(dataroot, './processed/SQuAD1.0/Graph_Analysis/SceneGraph/train/question_valid_node_label.npy'))
    all_valid_node_label = np.array(train_valid_node_label)

    print(len(all_answer_taggings), len(all_end_tagging), len(all_valid_node_label))
    print(len(all_vector_src), len(all_nodes_vector), len(all_edges_vector), len(all_nodes_type_vector), len(all_vector_tgt))

    all_vector_src, all_nodes_vector, all_edges_vector, all_adjacency, all_vector_tgt, all_questions, all_nodes_type_vector, all_vector_answer \
        = spread_vector(all_vector_src, all_nodes_vector, all_edges_vector, train_all_sent_adjacency,
                        all_vector_tgt, train_questions, is_path=True, all_nodes_type_vector=all_nodes_type_vector,
                        answer_vector=all_vector_answer)
    print(len(all_vector_src), len(all_nodes_vector), len(all_nodes_type_vector), len(all_edges_vector), len(all_vector_tgt), len(all_vector_answer))

    params = (batch_size, all_vector_src, all_nodes_vector,all_edges_vector, \
              all_adjacency, all_vector_tgt, all_answer_taggings, all_end_tagging,\
              all_path_tagging, all_neighbor_tagging, all_valid_node_label,\
              word_to_inx_src, word_to_inx_tgt, all_nodes_type_vector, all_vector_answer)

    return params


def get_path_train_data_each_epoch(params, is_shuffle=True):
    batch_size, all_vector_src, all_nodes_vector,all_edges_vector, \
    all_adjacency, all_vector_tgt, all_answer_taggings, all_end_tagging,\
    all_path_tagging, all_neighbor_tagging, all_valid_node_label,\
    word_to_inx_src, word_to_inx_tgt, all_nodes_type_vector, all_vector_answer, \
     = params

    index = np.arange(len(all_vector_src))
    if is_shuffle:
        print('Shuffle')
        np.random.shuffle(index)

    input_encode_batches, input_encode_lengths, input_nodes_batches, \
    input_edges_batches, input_adj_batches, input_answer_tagging_batches, \
    input_end_tagging_batches, input_path_tagging_batches, \
    input_neighbor_tagging_batches, valid_node_label_batches, input_decode_batches, \
    target_decode_batches, input_decode_lengths, input_nodes_type_batches, _, \
    input_answer_batches, input_answer_lengthes \
        = get_graph_batches(batch_size, all_vector_src, all_nodes_vector,
                            all_edges_vector, all_adjacency, all_vector_tgt,
                            all_answer_taggings, all_end_tagging,
                            all_path_tagging, all_neighbor_tagging, all_valid_node_label,
                            word_to_inx_src, word_to_inx_tgt, index=index,
                            all_nodes_type_vector=all_nodes_type_vector, all_vector_answer=all_vector_answer)

    return all_vector_src, all_vector_tgt, input_encode_batches, input_encode_lengths, \
           input_nodes_batches, input_edges_batches, input_adj_batches, \
           input_answer_tagging_batches, input_end_tagging_batches, \
           input_path_tagging_batches, input_neighbor_tagging_batches, valid_node_label_batches, \
           input_decode_batches, None, target_decode_batches, input_decode_lengths, \
           input_nodes_type_batches, input_answer_batches, input_answer_lengthes, None


def get_path_val_data(batch_size, vocabulary_src, vocabulary_tgt):
    _, val_questions, _, _ = importData(data_type=1)
    _, val_all_sent_nodes_type, _, val_all_sent_adjacency = import_graph_node_edge(data_type=1)

    word_to_inx_src = dict(zip(vocabulary_src, range(len(vocabulary_src))))
    word_to_inx_tgt = dict(zip(vocabulary_tgt, range(len(vocabulary_tgt))))

    all_vector_src, all_vector_answer, all_nodes_vector, all_nodes_type_vector, all_edges_vector, all_vector_tgt = save_import_vectors(is_save=False, is_path=True, data_type=1, is_answer=True)
    val_answer_taggings = np.load(os.path.join(dataroot, './processed/SQuAD1.0/val/answer_labels.npy'))
    all_answer_taggings = np.array(val_answer_taggings)
    all_neighbor_tagging = all_answer_taggings
    val_end_tagging = np.load(os.path.join(dataroot, './processed/SQuAD1.0/Graph_Analysis/SceneGraph/val/question_end_tagging.npy'))
    all_end_tagging = np.array(val_end_tagging)
    all_path_tagging = all_end_tagging
    val_valid_node_label = np.load(os.path.join(dataroot, './processed/SQuAD1.0/Graph_Analysis/SceneGraph/val/question_valid_node_label.npy'))
    all_valid_node_label = np.array(val_valid_node_label)

    all_vector_src, all_nodes_vector, all_edges_vector, all_adjacency, all_vector_tgt, all_questions, all_nodes_type_vector, all_vector_answer \
        = spread_vector(all_vector_src, all_nodes_vector, all_edges_vector,
                        val_all_sent_adjacency, all_vector_tgt, val_questions,
                        is_path=True, all_nodes_type_vector=all_nodes_type_vector,
                        answer_vector=all_vector_answer)
    print(len(all_vector_src), len(all_nodes_vector), len(all_nodes_type_vector), len(all_edges_vector), len(all_vector_tgt), len(all_vector_answer))

    input_encode_batches, input_encode_lengths, input_nodes_batches, \
    input_edges_batches, input_adj_batches, input_answer_tagging_batches, \
    input_end_tagging_batches, input_path_tagging_batches, \
    input_neighbor_tagging_batches, valid_node_label_batches, input_decode_batches, _, \
    input_decode_lengths, input_nodes_type_batches, _, input_answer_batches, input_answer_lengthes,\
        = get_graph_batches(batch_size, all_vector_src, all_nodes_vector,
                            all_edges_vector, all_adjacency, all_vector_tgt,
                            all_answer_taggings, all_end_tagging, all_path_tagging,
                            all_neighbor_tagging, all_valid_node_label,
                            word_to_inx_src, word_to_inx_tgt, all_nodes_type_vector=all_nodes_type_vector,
                            all_vector_answer=all_vector_answer)

    return all_vector_src, all_vector_tgt, input_encode_batches, input_encode_lengths, \
           input_nodes_batches, input_edges_batches, input_adj_batches, \
           input_answer_tagging_batches, input_end_tagging_batches, \
           input_path_tagging_batches, input_neighbor_tagging_batches, valid_node_label_batches,\
           input_decode_batches, all_questions, input_decode_lengths, input_nodes_type_batches, \
           input_answer_batches, input_answer_lengthes, None


def get_path_test_data(batch_size, vocabulary_src, vocabulary_tgt):
    test_sentences, test_questions, _, _ = importData(data_type=2)
    _, test_all_sent_nodes_type, _, test_all_sent_adjacency = import_graph_node_edge(data_type=2)

    word_to_inx_src = dict(zip(vocabulary_src, range(len(vocabulary_src))))
    word_to_inx_tgt = dict(zip(vocabulary_tgt, range(len(vocabulary_tgt))))

    all_vector_src, all_vector_answer, all_nodes_vector, all_nodes_type_vector, all_edges_vector, all_vector_tgt = save_import_vectors(is_save=False, is_path=True, data_type=2, is_answer=True)
    test_answer_taggings = np.load(os.path.join(dataroot, './processed/SQuAD1.0/test/answer_labels.npy'))
    all_answer_taggings = np.array(test_answer_taggings)
    all_neighbor_tagging = all_answer_taggings

    test_end_tagging = np.load(os.path.join(dataroot, './processed/SQuAD1.0/Graph_Analysis/SceneGraph/test/question_end_tagging.npy'))
    all_end_tagging = np.array(test_end_tagging)
    all_path_tagging = all_end_tagging
    test_valid_node_label = np.load(os.path.join(dataroot, './processed/SQuAD1.0/Graph_Analysis/SceneGraph/test/question_valid_node_label.npy'))
    all_valid_node_label = np.array(test_valid_node_label)

    all_vector_src, all_nodes_vector, all_edges_vector, all_adjacency, all_vector_tgt, all_questions, all_nodes_type_vector, all_vector_answer, all_sentences \
        = spread_vector(all_vector_src, all_nodes_vector, all_edges_vector, test_all_sent_adjacency,
                        all_vector_tgt, test_questions, is_path=True, all_nodes_type_vector=all_nodes_type_vector,
                        answer_vector=all_vector_answer, sentences=test_sentences)
    print(len(all_vector_src), len(all_nodes_vector), len(all_nodes_type_vector), len(all_edges_vector), len(all_vector_tgt), len(all_vector_answer))
    spread_spice_ref_questions = list()
    for i in range(len(test_questions)):
        for j in range(len(test_questions[i])):
            spread_spice_ref_questions.append(test_questions[i][j])

    input_encode_batches, input_encode_lengths, input_nodes_batches, \
    input_edges_batches, input_adj_batches, input_answer_tagging_batches, \
    input_end_tagging_batches, input_path_tagging_batches, \
    input_neighbor_tagging_batches, valid_node_label_batches, input_decode_batches, _, \
    input_decode_lengths, input_nodes_type_batches, _, input_answer_batches, input_answer_lengthes, \
        = get_graph_batches(batch_size, all_vector_src, all_nodes_vector,
                            all_edges_vector, all_adjacency, all_vector_tgt,
                            all_answer_taggings, all_end_tagging,
                            all_path_tagging, all_neighbor_tagging, all_valid_node_label,
                            word_to_inx_src, word_to_inx_tgt, all_nodes_type_vector=all_nodes_type_vector,
                            all_vector_answer=all_vector_answer)

    return all_vector_src, all_vector_tgt, input_encode_batches, input_encode_lengths, \
           input_nodes_batches, input_edges_batches, input_adj_batches, \
           input_answer_tagging_batches, input_end_tagging_batches, \
           input_path_tagging_batches, input_neighbor_tagging_batches, valid_node_label_batches, \
           input_decode_batches, all_questions, input_decode_lengths, input_nodes_type_batches, \
           input_answer_batches, spread_spice_ref_questions, all_sentences, None


def generate_graph_src_vocab(src_data, src_node_data, src_edge_data, fre_bound=None, src_node_type_data=None):
    # generate vocabulary for input text
    all_words = []
    for sentence in src_data:
        all_words += nltk.word_tokenize(sentence)
    for i in range(len(src_node_data)):
        for j in range(len(src_node_data[i])):
            all_words += src_node_data[i][j]
            for k in range(len(src_edge_data[i][j])):
                all_words += src_edge_data[i][j][k]
    counter = collections.Counter(all_words)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    words, fre = zip(*count_pairs)
    print(len(words))
    if fre_bound is not None:
        for i, each in enumerate(fre):
            if each == fre_bound:
                vocab_size = i
                break
        print(vocab_size)
        words = words[:vocab_size]
    words = ('PAD',) + words + ('UNK', 'S_token',)
    if src_node_type_data is not None:
        for i in range(len(src_node_type_data)):
            for j in range(len(src_node_type_data[i])):
                if src_node_type_data[i][j] not in words:
                    print(src_node_type_data[i][j])
                    words = words + (src_node_type_data[i][j],)
    return words


def generate_tgt_vocab(tgt_data_train, fre_bound=None):
    # generate vocabulary for question
    all_words = []
    for question_list in tgt_data_train:
        for question in question_list:
            all_words += nltk.word_tokenize(question)
    counter = collections.Counter(all_words)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    words, fre = zip(*count_pairs)
    print(len(words))
    if fre_bound is not None:
        for i, each in enumerate(fre):
            if each == fre_bound:
                vocab_size = i
                break
        print(vocab_size)
        words = words[:vocab_size]
    words = ('PAD',) + words + ('UNK',)
    return words


def get_position(data_type=0):
    # for answer focused and position aware model
    sentences, _, answers, answer_starts = importData(data_type=data_type)
    position_list = list()
    for i in tqdm(range(len(sentences))):
        sent_tokens = nltk.word_tokenize(sentences[i])
        for j in range(len(answers[i])):
            cur_ques_position_list = list()
            answer_location = len(nltk.word_tokenize(sentences[i][:answer_starts[i][j]]))
            for k, each_token in enumerate(sent_tokens):
                cur_ques_position_list.append(abs(k-answer_location))
            position_list.append(cur_ques_position_list)
    print(max([max(each) for each in position_list]))

    if data_type == 0:
        position_list_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/train/positions.npy'
    elif data_type == 1:
        position_list_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/val/positions.npy'
    else:
        position_list_file = './processed/SQuAD1.0/Graph_Analysis/SceneGraph/test/positions.npy'
    np.save(os.path.join(dataroot, position_list_file), position_list)


def replace_answer(sentences, answers):
    # for answer separation model
    replaced_sentences = list()
    for i in range(len(sentences)):
        cur_sentences = list()
        for j in range(len(answers[i])):
            cur_sent_tokens = nltk.word_tokenize(sentences[i])
            cur_answer_tokens = nltk.word_tokenize(answers[i][j])

            if answers[i][j] in sentences[i]:
                c_index = sentences[i].find(answers[i][j])
                w_index = len(nltk.word_tokenize(sentences[i][:c_index]))
                cur_sent_tokens[w_index: w_index + len(cur_answer_tokens)] = ['S_token', ]

            cur_sentences.append(' '.join(cur_sent_tokens))
        replaced_sentences.append(cur_sentences)
    return replaced_sentences


def analyse_graph():
    # analyse the constructed graph
    sent_nodes_train, _, sent_edges_train, _ = import_graph_node_edge(data_type=0)
    sent_nodes_val, _, sent_edges_val, _ = import_graph_node_edge(data_type=1)
    sent_nodes_test, _, sent_edges_test, _ = import_graph_node_edge(data_type=2)
    all_sent_nodes = sent_nodes_train.tolist() + sent_nodes_val.tolist() + sent_nodes_test.tolist()
    all_sent_edges = sent_edges_train.tolist() + sent_edges_val.tolist() + sent_edges_test.tolist()

    sentences_train, _, _, _ = importData(data_type=0)
    sentences_val, _, _, _ = importData(data_type=1)
    sentences_test, _, _, _ = importData(data_type=2)
    all_sentences = sentences_train.tolist() + sentences_val.tolist() + sentences_test.tolist()

    all_nodes = list()
    all_edges = list()

    all_sent_triple_numbers = list() # also relation number
    all_sent_entity_numbers = list()
    all_sent_overlap_rate = list()
    for i in tqdm(range(len(all_sent_nodes))):
        each_sent_nodes = all_sent_nodes[i]
        all_sent_entity_numbers.append(len(each_sent_nodes))

        each_sent_edges = []
        for j in range(len(all_sent_edges[i])):
            for each in all_sent_edges[i][j]:
                if len(each) > 0 and each != ['self-loop'] and each not in each_sent_edges:
                    each_sent_edges.append(each)
        all_sent_triple_numbers.append(len(each_sent_edges))

        overlap_token_number = 0
        for each_node in each_sent_nodes:
            all_nodes.append(' '.join(each_node))
            if ' '.join(each_node) in all_sentences[i]:
                overlap_token_number += len(each_node)
        for each_edges in each_sent_edges:
            all_edges.append(' '.join(each_edges))
            if ' '.join(each_edges) in all_sentences[i]:
                overlap_token_number += len(each_edges)
        all_sent_tokens = nltk.word_tokenize(all_sentences[i])
        overlap_rate = overlap_token_number / len(all_sent_tokens)
        all_sent_overlap_rate.append(overlap_rate)
    print('average quantity of facts', np.mean(all_sent_triple_numbers))
    print('average quantity of entities', np.mean(all_sent_entity_numbers))
    print('average coverage rate', np.mean(all_sent_overlap_rate))


def get_question_length(is_long=False):
    if is_long:
        train_sentences, train_questions, _, _, _, _, _, _ = get_long_data(data_type=0, is_save=False)
        val_sentences, val_questions, _, _, _, _, _, _ = get_long_data(data_type=1, is_save=False)
        test_sentences, test_questions, _, _, _, _, _, _ = get_long_data(data_type=2, is_save=False)
    else:
        train_sentences, train_questions, _, _ = importData(data_type=0)
        val_sentences, val_questions, _, _ = importData(data_type=1)
        test_sentences, test_questions, _, _ = importData(data_type=2)

    all_sentences = train_sentences.tolist() + val_sentences.tolist() + test_sentences.tolist()
    all_questions = train_questions.tolist() + val_questions.tolist() + test_questions.tolist()

    all_sentence_lengths = list()
    all_question_lengths = list()
    for i in range(len(all_sentences)):
        all_sentence_lengths.append(len(nltk.word_tokenize(all_sentences[i])))
        for j in range(len(all_questions[i])):
            all_question_lengths.append(len(nltk.word_tokenize(all_questions[i][j])))
    print(np.mean(all_sentence_lengths))
    print(np.mean(all_question_lengths))


if __name__ == '__main__':
    import_graph(data_type=0)
    import_graph(data_type=1)
    import_graph(data_type=2)

    # construct vocabulary of src and tgt
    train_sentences, train_questions, _, _ = importData(data_type=0)
    train_all_sent_nodes, train_all_sent_nodes_types, train_all_sent_edges, _ = import_graph_node_edge(data_type=0)
    print('Data loaded!')

    vocabulary_src = generate_graph_src_vocab(train_sentences, train_all_sent_nodes, train_all_sent_edges, fre_bound=2, src_node_type_data=train_all_sent_nodes_types)
    vocabulary_tgt = generate_tgt_vocab(train_questions, fre_bound=2)
    print(len(vocabulary_src), len(vocabulary_tgt))

    np.save(os.path.join(dataroot, './processed/SQuAD1.0/Graph_Analysis/SceneGraph/graph_vocabulary_src'), vocabulary_src)
    np.save(os.path.join(dataroot, './processed/SQuAD1.0/Graph_Analysis/SceneGraph/graph_vocabulary_tgt'), vocabulary_tgt)

    # generate vector of src, node, edge and tgt for baseline model
    save_import_vectors(is_save=True, data_type=0)
    save_import_vectors(is_save=True, data_type=1)
    save_import_vectors(is_save=True, data_type=2)

    find_end_path(data_type=0)
    find_end_path(data_type=1)
    find_end_path(data_type=2)
    end_path_tagging(data_type=0)
    end_path_tagging(data_type=1)
    end_path_tagging(data_type=2)
    extend_node_edge_to_path(data_type=0)
    extend_node_edge_to_path(data_type=1)
    extend_node_edge_to_path(data_type=2)

    # generate vector of src, node, edge and tgt for path-based model
    save_import_vectors(is_save=True, is_path=True, data_type=0)
    save_import_vectors(is_save=True, is_path=True, data_type=1)
    save_import_vectors(is_save=True, is_path=True, data_type=2)

    # Split Long Data and get long data index
    split_long_short_path_length_data(data_type=0)
    split_long_short_path_length_data(data_type=1)
    split_long_short_path_length_data(data_type=2)

    # # get ground truth path component tagging for upper bound model
    # get_valid_node_labels(data_type=0)
    # get_valid_node_labels(data_type=1)
    # get_valid_node_labels(data_type=2)
    # get_selected_path_tagging(data_type=0)
    # get_selected_path_tagging(data_type=1)
    # get_selected_path_tagging(data_type=2)

    # # get our selected path component tagging for pipeline model
    # get_selected_path_tagging_ours(data_type=0)
    # get_selected_path_tagging_ours(data_type=1)
    # get_selected_path_tagging_ours(data_type=2)

    # get position information for answer focused model
    # get_position(data_type=0)
    # get_position(data_type=1)
    # get_position(data_type=2)

    # analyse_graph()

    pass


