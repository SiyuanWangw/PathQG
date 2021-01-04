# for constructing KG
import sys
sys.path.append('..')
sys.path.append('../..')
from SceneGraphParser import sng_parser
from Preprocessing import *
from Graph_Structure import *
from tqdm import tqdm
import time
import pickle

start_time = time.time()
parser = sng_parser.Parser('spacy', model='en')


def construct(data_type=0):
    sentences, questions, answer_texts, answer_locations = importData(data_type=data_type)
    all_graphs = list()

    for i in tqdm(range(len(sentences))):
        sent = str(sentences[i])
        graph = Graph(sent, answer_texts[i], questions[i])

        graph_dict = parser.parse(sent)
        for each_entity in graph_dict['entities']:
            graph.extend_graph_node(each_entity['span'], None, each_entity['span_bounds'], each_entity['type'])
        for each_relation in graph_dict['relations']:
            subject_index = int(each_relation['subject'])
            object_index = int(each_relation['object'])
            graph.extend_graph_edge(each_relation['relation'], each_relation['lemma_relation'],
                                    graph.node_list[subject_index], graph.node_list[object_index])
        all_graphs.append(graph)

    if data_type == 0:
        graph_file_name = '../../Data/processed/SQuAD1.0/Graph_Analysis/train/scene_graphs.pkl'
    elif data_type == 1:
        graph_file_name = '../../Data/processed/SQuAD1.0/Graph_Analysis/val/scene_graphs.pkl'
    else:
        graph_file_name = '../../Data/processed/SQuAD1.0/Graph_Analysis/test/scene_graphs.pkl'

    fw = open(graph_file_name, 'wb')
    pickle.dump(all_graphs, fw)
    fw.close()


def reverse_index(l, obj):
    n = len(l)
    while n > 0:
        n = n - 1
        if l[n] == obj:
           return n

def reverse_ith_index(l, obj, ith):
    n = len(l)
    i = 0
    while n > 0:
        n = n - 1
        if l[n] == obj:
            i += 1
        if i == ith:
            return n

    if i < ith:
        print('Can not find the ith index!')


def extract_relation(sent_words, word_pos_taggings):
    verb_or_preposition = list()
    conj = list()
    for i in range(len(word_pos_taggings)):
        if word_pos_taggings[i][0] == '-lrb-' or word_pos_taggings[i][0] == '-rrb-':
            conj.append((i, word_pos_taggings[i]))
        if word_pos_taggings[i][1] == 'ADP' or word_pos_taggings[i][1] == 'PRT' or word_pos_taggings[i][1] == 'VERB':
            if word_pos_taggings[i][0] != '-lrb-' and word_pos_taggings[i][0] != '-rrb-':
                verb_or_preposition.append((i, word_pos_taggings[i]))
        elif word_pos_taggings[i][0] == ',' or word_pos_taggings[i][0] == 'or' or word_pos_taggings[i][0] == 'and':
            conj.append((i, word_pos_taggings[i]))

    if len(verb_or_preposition) == 0:
        if len(conj) == 0:
            # need to add null edge???
            # relationText = 'null'
            # relationType = 'null'
            return
        else:
            if len(conj) == 1:
                relationText = conj[0][1][0]
            elif len(conj) == 2 and (conj[1][0] - conj[0][0]) == 1 and conj[0][1][0] != ',' and conj[1][1][0] != ',':
                relationText = ' '.join([conj[0][1][0], conj[1][1][0]])
            elif len(conj) == 2 and conj[0][1][0] != ',' and conj[1][1][0] != ',':
                relationText = conj[1][1][0]
            elif len(conj) == 2 and conj[0][1][0] == ',':
                relationText = conj[1][1][0]
            elif len(conj) == 2 and conj[1][1][0] == ',':
                relationText = conj[0][1][0]
            elif len(conj) > 2:
                relationText = conj[-1][1][0]
            else:
                print('-----1-----', sent_words)
                return
    elif len(verb_or_preposition) == 1:
        relationText = verb_or_preposition[0][1][0]
    elif len(verb_or_preposition) == 2:
        if verb_or_preposition[0][1][1] == 'VERB' and verb_or_preposition[1][1][1] == 'PRT':
            relationText = ' '.join(sent_words[verb_or_preposition[0][0]: verb_or_preposition[1][0]+1])
        elif verb_or_preposition[0][1][1] == 'PRT' and verb_or_preposition[1][1][1] == 'VERB':
            relationText = ' '.join(sent_words[verb_or_preposition[0][0]: verb_or_preposition[1][0] + 1])
        elif verb_or_preposition[0][1][1] == 'VERB' and verb_or_preposition[1][1][1] == 'ADP':
            if (verb_or_preposition[1][0]-verb_or_preposition[0][0]) == 1:
                relationText = ' '.join([verb_or_preposition[0][1][0], verb_or_preposition[1][1][0]])
            elif (verb_or_preposition[1][0]-verb_or_preposition[0][0]) == 2:
                relationText = ' '.join(sent_words[verb_or_preposition[0][0]: verb_or_preposition[1][0]+1])
            else:
                relationText = verb_or_preposition[0][1][0]
        elif verb_or_preposition[0][1][1] == 'ADP' and verb_or_preposition[1][1][1] == 'VERB':
            if (verb_or_preposition[1][0]-verb_or_preposition[0][0]) == 1:
                relationText = ' '.join([verb_or_preposition[0][1][0], verb_or_preposition[1][1][0]])
            else:
                relationText = verb_or_preposition[1][1][0]
        elif verb_or_preposition[0][1][1] == 'VERB' and verb_or_preposition[1][1][1] == 'VERB' and (verb_or_preposition[1][0]-verb_or_preposition[0][0]) == 1:
            relationText = ' '.join([verb_or_preposition[0][1][0], verb_or_preposition[1][1][0]])
        elif verb_or_preposition[0][1][1] == 'VERB' and verb_or_preposition[1][1][1] == 'VERB':
            relationText = verb_or_preposition[1][1][0]
        elif verb_or_preposition[0][1][1] == 'ADP' and verb_or_preposition[1][1][1] == 'ADP':
            relationText = verb_or_preposition[1][1][0]
        elif verb_or_preposition[0][1][1] == 'PRT' and verb_or_preposition[1][1][1] == 'ADP':
            relationText = verb_or_preposition[1][1][0]
        elif verb_or_preposition[0][1][1] == 'ADP' and verb_or_preposition[1][1][1] == 'PRT':
            relationText = verb_or_preposition[0][1][0]
        elif verb_or_preposition[0][1][1] == 'PRT' and verb_or_preposition[1][1][1] == 'PRT':
            relationText = ' '.join(sent_words[verb_or_preposition[0][0]: verb_or_preposition[1][0] + 1])
        else:
            print('-----2-----', sent_words)
            return
    else:
        pos_list = [each[1][1] for each in verb_or_preposition]
        if pos_list.count('VERB') == 1:
            if pos_list.count('PRT') == 0:
                if pos_list.count('ADP') == 0:
                    relationText = verb_or_preposition[pos_list.index('VERB')][1][0]
                elif pos_list.count('ADP') == 1:
                    ADP_index = pos_list.index('ADP')
                    VERB_index = pos_list.index('VERB')
                    if verb_or_preposition[ADP_index][0] - verb_or_preposition[VERB_index][0] == 1:
                        relationText = ' '.join([verb_or_preposition[VERB_index][1][0], verb_or_preposition[ADP_index][1][0]])
                    elif verb_or_preposition[ADP_index][0] - verb_or_preposition[VERB_index][0] == -1:
                        relationText = ' '.join([verb_or_preposition[ADP_index][1][0], verb_or_preposition[VERB_index][1][0]])
                    else:
                        relationText = verb_or_preposition[VERB_index][1][0]
                else: # pos_list.count('ADP') >= 2:
                    VERB_index = pos_list.index('VERB')
                    ADP_index_1 = reverse_ith_index(pos_list, 'ADP', 2)
                    ADP_index_2 = reverse_index(pos_list, 'ADP')
                    if (verb_or_preposition[VERB_index][0] - verb_or_preposition[ADP_index_1][0]) == 1 and \
                        (verb_or_preposition[ADP_index_2][0] - verb_or_preposition[VERB_index][0]) == 1:
                        relationText = ' '.join([verb_or_preposition[ADP_index_1][1][0], verb_or_preposition[VERB_index][1][0],
                                                 verb_or_preposition[ADP_index_2][1][0]])
                    elif (verb_or_preposition[VERB_index][0] - verb_or_preposition[ADP_index_1][0]) == 1:
                        relationText = ' '.join([verb_or_preposition[ADP_index_1][1][0], verb_or_preposition[VERB_index][1][0]])
                    elif (verb_or_preposition[VERB_index][0] - verb_or_preposition[ADP_index_2][0]) == 1:
                        relationText = ' '.join([verb_or_preposition[ADP_index_2][1][0], verb_or_preposition[VERB_index][1][0]])
                    elif (verb_or_preposition[VERB_index][0] - verb_or_preposition[ADP_index_1][0]) == -1:
                        relationText = ' '.join([verb_or_preposition[VERB_index][1][0], verb_or_preposition[ADP_index_1][1][0]])
                    elif (verb_or_preposition[VERB_index][0] - verb_or_preposition[ADP_index_2][0]) == -1:
                        relationText = ' '.join([verb_or_preposition[VERB_index][1][0], verb_or_preposition[ADP_index_2][1][0]])
                    else:
                        relationText = verb_or_preposition[VERB_index][1][0]
            elif pos_list.count('PRT') == 1:
                start_index = pos_list.index('VERB')
                end_index = pos_list.index('PRT')
                relationText = ' '.join(sent_words[verb_or_preposition[start_index][0]: verb_or_preposition[end_index][0]+1])
            elif pos_list.count('PRT') == 2:
                start_index = min(pos_list.index('VERB'), pos_list.index('PRT'))
                end_index = reverse_index(pos_list, 'PRT')
                relationText = ' '.join(sent_words[verb_or_preposition[start_index][0]: verb_or_preposition[end_index][0] + 1])
            else: #pos_list.count('PRT') > 2:
                if verb_or_preposition[reverse_index(pos_list, 'PRT')][1][0] == "'":
                    start_index = min(pos_list.index('VERB'), pos_list.index('PRT'))
                    end_index = reverse_ith_index(pos_list, 'PRT', 2)
                    relationText = ' '.join(
                        sent_words[verb_or_preposition[start_index][0]: verb_or_preposition[end_index][0] + 1])
                else:
                    start_index = min(pos_list.index('PRT'), pos_list.index('VERB'))
                    end_index = reverse_index(pos_list, 'PRT')
                    relationText = ' '.join(
                        sent_words[verb_or_preposition[start_index][0]: verb_or_preposition[end_index][0] + 1])
        elif pos_list.count('VERB') > 1:
            if pos_list.count('PRT') == 0 and pos_list.count('ADP') == 0:
                end_index = reverse_index(pos_list, 'VERB')
                relationText = verb_or_preposition[end_index][1][0]
            elif pos_list.count('PRT') >= 1 and pos_list.count('ADP') == 0:
                start_index = reverse_index(pos_list, 'VERB')
                end_index = max(reverse_index(pos_list, 'VERB'), reverse_index(pos_list, 'PRT'))
                relationText = ' '.join(sent_words[verb_or_preposition[start_index][0]: verb_or_preposition[end_index][0] + 1])
            elif pos_list.count('PRT') == 0 and pos_list.count('ADP') >= 1:
                start_index = reverse_index(pos_list, 'VERB')
                end_index = max(reverse_index(pos_list, 'VERB'), reverse_index(pos_list, 'ADP'))
                relationText = ' '.join(
                    sent_words[verb_or_preposition[start_index][0]: verb_or_preposition[end_index][0] + 1])
            elif pos_list.count('PRT') >= 1 and pos_list.count('ADP') >= 1:
                start_index = reverse_index(pos_list, 'VERB')
                end_index = max(reverse_index(pos_list, 'PRT'), reverse_index(pos_list, 'ADP'), reverse_index(pos_list, 'VERB'))
                relationText = ' '.join(
                    sent_words[verb_or_preposition[start_index][0]: verb_or_preposition[end_index][0] + 1])
            else:
                print('-----4-----', sent_words)
                return
        else: # pos_list.count('VERB') == 0
            if pos_list.count('PRT') == 0:
                start_index = reverse_index(pos_list, 'ADP')
                end_index = start_index
                relationText = ' '.join(
                    sent_words[verb_or_preposition[start_index][0]: verb_or_preposition[end_index][0] + 1])
            elif pos_list.count('ADP') == 0:
                start_index = reverse_index(pos_list, 'PRT')
                end_index = start_index
                relationText = ' '.join(
                    sent_words[verb_or_preposition[start_index][0]: verb_or_preposition[end_index][0] + 1])
            else:
                start_index = max(reverse_index(pos_list, 'PRT'), reverse_index(pos_list, 'ADP'))
                end_index = start_index
                relationText = ' '.join(
                    sent_words[verb_or_preposition[start_index][0]: verb_or_preposition[end_index][0] + 1])
    return relationText, relationText


def enrich_graphs(data_type=0):
    if data_type == 0:
        graph_file_name = '../../Data/processed/SQuAD1.0/Graph_Analysis/train/scene_graphs.pkl'
    elif data_type == 1:
        graph_file_name = '../../Data/processed/SQuAD1.0/Graph_Analysis/val/scene_graphs.pkl'
    else:
        graph_file_name = '../../Data/processed/SQuAD1.0/Graph_Analysis/test/scene_graphs.pkl'

    fr = open(graph_file_name, 'rb')
    graphs = pickle.load(fr)
    fr.close()

    total_added_edges_num = 0
    sentences, questions, _, _ = importData(data_type=data_type)

    for i in tqdm(range(len(graphs))):
        each_graph = graphs[i]
        sent_words_list = nltk.word_tokenize(sentences[i])
        pos_taggings = nltk.pos_tag(sent_words_list, tagset='universal')
        cur_edge_num = len(each_graph.edge_list)

        for j, each_graph_node in enumerate(each_graph.node_list):
            cur_node_span_bounds = each_graph_node.nodespan_bounds

            # enrich edges
            if j == 0 and (each_graph_node.nodetype == 'DATE' or each_graph_node.nodetype == 'TIME'):
                if j+1 >= len(each_graph.node_list):
                    pass
                else:
                    if each_graph.get_edge_by_index(j+1, j) is None:
                        end_index = cur_node_span_bounds[0]
                        relation = extract_relation(sent_words_list[: end_index], pos_taggings[: end_index])
                        if relation is not None:
                            relationText, relationType = relation
                            each_graph.extend_graph_edge(relationText, relationType, each_graph.node_list[j+1], each_graph.node_list[j])
            elif j < len(each_graph.node_list) - 1:
                if each_graph.get_edge_by_index(j, j+1) is None:
                    if '-' not in sentences[i]:
                        start_index = cur_node_span_bounds[1]
                        end_index = each_graph.node_list[j+1].nodespan_bounds[0]
                    else:
                        cur_node_text = each_graph_node.nodetext
                        start_index = len(nltk.word_tokenize(sentences[i][:sentences[i].find(cur_node_text)])) + len(nltk.word_tokenize(cur_node_text))
                        end_index = len(nltk.word_tokenize(sentences[i][:sentences[i].find(each_graph.node_list[j+1].nodetext)]))
                    relation = extract_relation(sent_words_list[start_index: end_index],
                                                                  pos_taggings[start_index: end_index])
                    if relation is not None:
                        # print(j, relation)
                        relationText, relationType = relation
                        each_graph.extend_graph_edge(relationText, relationType, each_graph.node_list[j], each_graph.node_list[j+1])

        total_added_edges_num += len(each_graph.edge_list) - cur_edge_num

    print(total_added_edges_num)

    if data_type == 0:
        graph_file_name = '../../Data/processed/SQuAD1.0/Graph_Analysis/train/enriched_scene_graphs.pkl'
    elif data_type == 1:
        graph_file_name = '../../Data/processed/SQuAD1.0/Graph_Analysis/val/enriched_scene_graphs.pkl'
    else:
        graph_file_name = '../../Data/processed/SQuAD1.0/Graph_Analysis/test/enriched_scene_graphs.pkl'

    fw = open(graph_file_name, 'wb')
    pickle.dump(graphs, fw)
    fw.close()


if __name__ == '__main__':
    construct(data_type=0)
    construct(data_type=1)
    construct(data_type=2)

    enrich_graphs(data_type=0)
    enrich_graphs(data_type=1)
    enrich_graphs(data_type=2)

    pass
