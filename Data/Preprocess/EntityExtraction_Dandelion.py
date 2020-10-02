# extract entities using dandelion API
import requests
from Preprocessing import *
import numpy as np
from tqdm import tqdm
import time
import nltk
import spacy
nlp = spacy.load('en')

start = time.time()
url = 'https://api.dandelion.eu/datatxt/nex/v1'
token = '' # to sign up for token

def extract_one_text(text):
    payload = {'text': text, 'lang': 'en', 'min_confidence': '0.0', 'token': token, 'include': 'types'}
    r = requests.get(url, params=payload)
    if r.status_code == 200:
        result = (True, r.json())
    else:
        result = (False, r.json())
    return result


def extract_all_data(data_type=0):
    sentences, _, _, _ = importData(data_type=data_type)
    all_annotations = list()
    error_num = 0
    start_index = 1000
    end_index = 2000

    for each_sent in tqdm(sentences[start_index:end_index]):
        cur_annotation = extract_one_text(each_sent)
        if not cur_annotation[0]:
            error_num += 1
        all_annotations.append(cur_annotation)

    print(error_num)
    if data_type == 0:
        np.save('../processed/SQuAD1.0/train/raw_entity_annotations_2', all_annotations)
    elif data_type == 1:
        np.save('../processed/SQuAD1.0/val/raw_entity_annotations_7', all_annotations)
    else:
        np.save('../processed/SQuAD1.0/test/raw_entity_annotations_7', all_annotations)


def combine_all_entity_file(data_type=0):
    if data_type == 0:
        file_name_prefix = '../processed/SQuAD1.0/train/raw_entity_annotations_'
    elif data_type == 1:
        file_name_prefix = '../processed/SQuAD1.0/val/raw_entity_annotations_'
    else:
        file_name_prefix = '../processed/SQuAD1.0/test/raw_entity_annotations_'

    all_annotations = list()
    file_number = 49 #7
    for i in range(file_number):
        print(i)
        file_name = file_name_prefix + str(i+1) + '.npy'
        content = np.load(file_name).tolist()
        all_annotations = all_annotations + content

    print(len(all_annotations))

    if data_type == 0:
        np.save('../processed/SQuAD1.0/train/raw_entity_annotations', all_annotations)
    elif data_type == 1:
        np.save('../processed/SQuAD1.0/val/raw_entity_annotations', all_annotations)
    else:
        np.save('../processed/SQuAD1.0/test/raw_entity_annotations', all_annotations)


def get_entity_location_type(data_type=0):
    all_entities = list()
    all_entities_location = list()
    all_entities_type = list()

    if data_type == 0:
        load_file = '../processed/SQuAD1.0/train/raw_entity_annotations.npy'
    elif data_type == 1:
        load_file = '../processed/SQuAD1.0/val/raw_entity_annotations.npy'
    else:
        load_file = '../processed/SQuAD1.0/test/raw_entity_annotations.npy'

    all_annotations = np.load(load_file)
    print(all_annotations[0])
    print(all_annotations[1])
    print(all_annotations[2])

    discarded_pos = ['ADJ', 'ADV', 'VERB']
    discarded_num = 0
    for _, each_content in tqdm(all_annotations):
        cur_sent_entities = list()
        cur_sent_entities_location = list()
        cur_sent_entities_type = list()
        cur_annotation = each_content['annotations']
        for each_entity_detail in cur_annotation:
            cur_entitiy = each_entity_detail['spot']
            entity_words = nltk.word_tokenize(cur_entitiy)
            entity_words_pos = [each[1] for each in nltk.pos_tag(entity_words, tagset='universal')]
            if len(entity_words) == 1 and entity_words_pos[0] in discarded_pos:
                discarded_num += 1
            elif len(entity_words) == 1 and entity_words_pos[0] == 'NUM':
                cur_sent_entities.append(cur_entitiy)
                cur_sent_entities_location.append((each_entity_detail['start'], each_entity_detail['end']))
                cur_sent_entities_type.append('NUMBER')
            elif len(entity_words) == 1 and entity_words_pos[0] == 'PRON':
                cur_sent_entities.append(cur_entitiy)
                cur_sent_entities_location.append((each_entity_detail['start'], each_entity_detail['end']))
                cur_sent_entities_type.append('PERSON')
            elif len(entity_words) > 1 and ('NOUN' not in entity_words_pos) and ('NUM' not in entity_words_pos) and ('PRON' not in entity_words_pos) and ('DET' not in entity_words_pos):
                discarded_num += 1
            else:
                cur_sent_entities.append(cur_entitiy)
                cur_sent_entities_location.append((each_entity_detail['start'], each_entity_detail['end']))
                if 'http://dbpedia.org/ontology/Person' in each_entity_detail['types']:
                    cur_sent_entities_type.append('PERSON')
                elif 'http://dbpedia.org/ontology/Organisation' in each_entity_detail['types']:
                    cur_sent_entities_type.append('ORGANIZATION')
                elif 'http://dbpedia.org/ontology/Location' in each_entity_detail['types']:
                    cur_sent_entities_type.append('LOCATION')
                elif 'http://dbpedia.org/ontology/Currency' in each_entity_detail['types']:
                    cur_sent_entities_type.append('MONEY')
                elif 'http://dbpedia.org/ontology/MeanOfTransportation' in each_entity_detail['types']:
                    cur_sent_entities_type.append('TRANSPORTATION')
                elif 'http://dbpedia.org/ontology/Holiday' in each_entity_detail['types']:
                    cur_sent_entities_type.append('HOLIDAY')
                elif 'http://dbpedia.org/ontology/Event' in each_entity_detail['types']:
                    cur_sent_entities_type.append('EVENT')
                elif 'http://dbpedia.org/ontology/Work' in each_entity_detail['types']:
                    cur_sent_entities_type.append('WORK')
                elif len(each_entity_detail['types']) == 0:
                    cur_sent_entities_type.append('CONCEPT')
                else:
                    cur_sent_entities_type.append('CONCEPT')
            # Language, EthnicGroup, Award
        all_entities.append(cur_sent_entities)
        all_entities_location.append(cur_sent_entities_location)
        all_entities_type.append(cur_sent_entities_type)

    print(len(all_entities))
    print(sum([len(each) for each in all_entities]))
    print('discarded number', discarded_num)
    save_entity_data(all_entities, all_entities_location, all_entities_type, data_type=data_type)


def get_np_by_spacy(data_type=0):
    sentences, _, _, _ = importData(data_type=data_type)
    spacy_nps = list()

    for sent in tqdm(sentences):
        doc = nlp(str(sent))
        spacy_nps.append([(each_np.text, each_np.root.head.text) for each_np in doc.noun_chunks])

    if data_type == 0:
        spacy_np_file = '../processed/SQuAD1.0/train/spacy_nps.npy'
    elif data_type == 1:
        spacy_np_file = '../processed/SQuAD1.0/val/spacy_nps.npy'
    else:
        spacy_np_file = '../processed/SQuAD1.0/test/spacy_nps.npy'
    np.save(spacy_np_file, spacy_nps)


def correct_entity_by_np(data_type=0):
    sentences, questions, answers, _ = importData(data_type=data_type)
    entities, entities_location, entities_type = import_entity_data(data_type=data_type)
    if data_type == 0:
        spacy_np_file = '../processed/SQuAD1.0/train/spacy_nps.npy'
    elif data_type == 1:
        spacy_np_file = '../processed/SQuAD1.0/val/spacy_nps.npy'
    else:
        spacy_np_file = '../processed/SQuAD1.0/test/spacy_nps.npy'
    spacy_nps = np.load(spacy_np_file)

    for i, entity in enumerate(entities):
        # for j, each_entities in enumerate(entity):
        print(sentences[i])
        print(questions[i])
        print(answers[i])
        print(entity)
        print(spacy_nps[i])
        print("#"*100)

        if i == 10:
            break


def location_has_overlap(start1, end1, locations):
    for i, (start2, end2) in enumerate(locations):
        if len(set(range(start1, end1)) & set(range(start2, end2))) > 0:
            return i
    return -1


def find_insert_index(start1, locations):
    insert_index = 0
    for i in range(1, len(locations)):
        if start1 > locations[i-1][0] and start1 <= locations[i][0]:
            insert_index = i
    if len(locations) > 0 and start1 > locations[-1][0]:
        insert_index = len(locations)
    return insert_index


def remove_lrb_rrb(data_type=0):
    rb_signals = ['-lrb-', '-rrb-', '-lrb', '-rrb', 'lrb-', 'rrb-','lrb', 'rrb']
    entities, entities_location, entities_type = import_entity_data(data_type=data_type)
    new_entities, new_entities_location, new_entities_type = list(), list(), list()
    remove_rb_num = 0
    for i, cur_sent_entities in enumerate(entities):
        new_sent_entities, new_sent_entities_location, new_sent_entities_type = list(), list(), list()
        for j, entity in enumerate(cur_sent_entities):
            if entity not in rb_signals:

                entity_tokens = nltk.word_tokenize(entity)
                has_rb_signal = False
                if rb_signals[0] in entity_tokens:
                    exist_rb_signal = rb_signals[0]
                    has_rb_signal = True
                elif rb_signals[1] in entity_tokens:
                    exist_rb_signal = rb_signals[1]
                    has_rb_signal = True
                elif rb_signals[2] in entity_tokens:
                    exist_rb_signal = rb_signals[2]
                    has_rb_signal = True
                elif rb_signals[3] in entity_tokens:
                    exist_rb_signal = rb_signals[3]
                    has_rb_signal = True
                elif rb_signals[4] in entity_tokens:
                    exist_rb_signal = rb_signals[4]
                    has_rb_signal = True
                elif rb_signals[5] in entity_tokens:
                    exist_rb_signal = rb_signals[5]
                    has_rb_signal = True
                elif rb_signals[6] in entity_tokens:
                    exist_rb_signal = rb_signals[6]
                    has_rb_signal = True
                elif rb_signals[7] in entity_tokens:
                    exist_rb_signal = rb_signals[7]
                    has_rb_signal = True

                if has_rb_signal:
                    remove_rb_num += 1
                    rb_index = entity_tokens.index(exist_rb_signal)
                    if rb_index == 0:
                        new_sent_entities.append(' '.join(entity_tokens[1:]))
                        new_sent_entities_location.append((entities_location[i][j][0]+len(exist_rb_signal)+1, entities_location[i][j][1]))
                        new_sent_entities_type.append(entities_type[i][j])
                    elif rb_index == len(entity_tokens)-1:
                        new_sent_entities.append(' '.join(entity_tokens[:-1]))
                        new_sent_entities_location.append((entities_location[i][j][0], entities_location[i][j][1]-len(exist_rb_signal) - 1))
                        new_sent_entities_type.append(entities_type[i][j])
                    else:
                        new_sent_entities.append(' '.join(entity_tokens[:rb_index]))
                        new_sent_entities_location.append((entities_location[i][j][0], entities_location[i][j][0]+len(new_sent_entities[-1])))
                        new_sent_entities_type.append(entities_type[i][j])

                        new_sent_entities.append(' '.join(entity_tokens[rb_index+1:]))
                        new_sent_entities_location.append((entities_location[i][j][1]-len(new_sent_entities[-1]), entities_location[i][j][1]))
                        new_sent_entities_type.append(entities_type[i][j])
                else:
                    new_sent_entities.append(entity)
                    new_sent_entities_location.append(entities_location[i][j])
                    new_sent_entities_type.append(entities_type[i][j])

            else:
                pass
        new_entities.append(new_sent_entities)
        new_entities_location.append(new_sent_entities_location)
        new_entities_type.append(new_sent_entities_type)

        if i % 5000 == 0:
            print(i)

    print('total length', len(new_entities))
    print('remove rb num', remove_rb_num)

    save_entity_data(new_entities, new_entities_location, new_entities_type, data_type=data_type)


def check_entities_NER(data_type=0):
    sentences, _, _, _ = importData(data_type=data_type)
    entities, entities_location, entities_type = import_entity_data(data_type=data_type)

    # DATE, TIME, (CARDINAL, ORDINAL)-NUMBER, NORP, GPE(if LANGUAGE needed)
    add_ner_number = 0
    type_set = ['DATE', 'TIME', 'NORP', 'QUANTITY', 'PERCENT', 'MONEY', 'PERSON', 'EVENT']
    month_words = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']
    discarded_num = 0
    discarded_pos = ['ADJ', 'ADV', 'VERB']
    for i, sent in enumerate(sentences):
        doc = nlp(str(sent))
        for e in doc.ents:
            if e.label_ == 'CARDINAL' or e.label_ == 'ORDINAL':
                type = 'NUMBER'
            elif e.label_ == 'GPE':
                type = 'LOCATION'
            elif e.label_ == 'ORG':
                type = 'ORGANIZATION'
            elif e.label_ == 'LOC':
                type = 'LOCATION'
            elif e.label_ in type_set:
                type = e.label_
            else:
                # print(e.text, e.label_)
                continue

            add_ner_number += 1

            overlap_index = location_has_overlap(int(e.start_char), int(e.end_char), entities_location[i])
            if overlap_index >= 0:
                entities[i].pop(overlap_index)
                entities[i].insert(overlap_index, e.text)
                entities_location[i].pop(overlap_index)
                entities_location[i].insert(overlap_index, (int(e.start_char), int(e.end_char)))
                entities_type[i].pop(overlap_index)
                entities_type[i].insert(overlap_index, type)
            else:
                entity_words = nltk.word_tokenize(e.text)
                if len(entity_words) == 1 and len(e.text.split('-')) == 1 and nltk.pos_tag(entity_words, tagset='universal')[0][1] in discarded_pos and e.text not in month_words:
                    discarded_num += 1
                    if type == 'DATE':
                        print(e.text, type)
                else:
                    insert_index = find_insert_index(int(e.start_char), entities_location[i])
                    entities[i].insert(insert_index, e.text)
                    entities_location[i].insert(insert_index, (int(e.start_char), int(e.end_char)))
                    entities_type[i].insert(insert_index, type)

        if i % 1000 == 0:
            print(i)

    print('number of ner added', add_ner_number)
    print('discard number', discarded_num)
    save_entity_data(entities, entities_location, entities_type, data_type=data_type)


def remove_no_noun_phrases(data_type=0):
    entities, entities_location, entities_type = import_entity_data(data_type=data_type)
    new_entities, new_entities_location, new_entities_type = list(), list(), list()
    discarded_pos = ['ADJ', 'ADV', 'VERB']
    month_words = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
                   'november', 'december']
    discarded_num = 0
    for i, cur_sent_entities in enumerate(entities):
        new_sent_entities, new_sent_entities_location, new_sent_entities_type = list(), list(), list()
        for j, entity in enumerate(cur_sent_entities):
            entity_words = nltk.word_tokenize(entity)
            entity_words_pos = [each[1] for each in nltk.pos_tag(entity_words, tagset='universal')]

            if len(entity_words) == 1 and (entity_words_pos[0] in discarded_pos or entity == 'versus') and entity not in month_words:
                discarded_num += 1
            elif len(entity_words) > 1 and ('NOUN' not in entity_words_pos) and ('NUM' not in entity_words_pos) and (
                    'PRON' not in entity_words_pos) and ('DET' not in entity_words_pos):
                discarded_num += 1
            else:
                new_sent_entities.append(entity)
                new_sent_entities_location.append(entities_location[i][j])
                new_sent_entities_type.append(entities_type[i][j])

        new_entities.append(new_sent_entities)
        new_entities_location.append(new_sent_entities_location)
        new_entities_type.append(new_sent_entities_type)

        if i % 1000 == 0:
            print(i)

    print('discard number', discarded_num)
    save_entity_data(new_entities, new_entities_location, new_entities_type, data_type=data_type)


## 暂且不使用
def split_conj_entities(data_type=0):
    entities, entities_location, entities_type = import_entity_data(data_type=data_type)
    new_entities, new_entities_location, new_entities_type = list(), list(), list()

    for i, cur_sent_entities in enumerate(entities):
        new_sent_entities, new_sent_entities_location, new_sent_entities_type = list(), list(), list()
        for j, entity in enumerate(cur_sent_entities):
            split_entities = entity.split(' , ')
            if len(split_entities) == 1:
                new_sent_entities.append(entity)
                new_sent_entities_location.append(entities_location[i][j])
                new_sent_entities_type.append(entities_type[i][j])
            else:
                start_loc = entities_location[i][j][0]
                for k, s_entity in enumerate(split_entities):
                    end_loc = start_loc + len(s_entity)
                    new_sent_entities.append(s_entity)
                    new_sent_entities_location.append((start_loc, end_loc))
                    new_sent_entities_type.append(entities_type[i][j])
                    start_loc = end_loc + 3

        new_entities.append(new_sent_entities)
        new_entities_location.append(new_sent_entities_location)
        new_entities_type.append(new_sent_entities_type)

    save_entity_data(new_entities, new_entities_location, new_entities_type, data_type=data_type)


if __name__ == '__main__':
    # extract_all_data(data_type=0)
    # extract_all_data(data_type=2)
    # print(time.time()-start)

    # combine_all_entity_file(data_type=0)
    # combine_all_entity_file(data_type=1)
    # combine_all_entity_file(data_type=2)

    # get_entity_location_type(data_type=0)
    # get_entity_location_type(data_type=1)
    # get_entity_location_type(data_type=2)

    # check_entities_NER(data_type=0)
    # check_entities_NER(data_type=1)
    # check_entities_NER(data_type=2)

    # remove_lrb_rrb(data_type=0)
    # remove_lrb_rrb(data_type=0)
    # remove_lrb_rrb(data_type=0)
    # remove_lrb_rrb(data_type=1)
    # remove_lrb_rrb(data_type=1)
    # remove_lrb_rrb(data_type=2)
    # remove_lrb_rrb(data_type=2)
    #
    # remove_no_noun_phrases(data_type=0)
    # remove_no_noun_phrases(data_type=1)
    # remove_no_noun_phrases(data_type=2)
    #
    # # process_question_entities(data_type=0)
    # # process_question_entities(data_type=1)
    # # process_question_entities(data_type=2)

    pass











