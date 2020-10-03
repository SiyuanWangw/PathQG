import numpy as np
import nltk
import os
from nltk.corpus import stopwords
dataroot = os.path.abspath('../../Data')  # graph

def importData(data_type=0, is_rm=True):
    if data_type == 0:
        ## training data
        files = (
            './processed/SQuAD1.0/train/sentences.npz', './processed/SQuAD1.0/train/questions.npy',
            './processed/SQuAD1.0/train/answers.npy', './processed/SQuAD1.0/train/answers_start.npy')
    elif data_type == 1:
        ## validation date
        files = (
            './processed/SQuAD1.0/val/sentences.npz', './processed/SQuAD1.0/val/questions.npy',
            './processed/SQuAD1.0/val/answers.npy', './processed/SQuAD1.0/val/answers_start.npy')
    else:
        ## test data
        if not is_rm:
            files = (
                './processed/SQuAD1.0/test/sentences.npz', './processed/SQuAD1.0/test/questions.npy',
                './processed/SQuAD1.0/test/answers.npy', './processed/SQuAD1.0/test/answers_start.npy')
        else:
            files = (
                './processed/SQuAD1.0/test/rm_sentences.npz', './processed/SQuAD1.0/test/rm_questions.npy',
                './processed/SQuAD1.0/test/rm_answers.npy', './processed/SQuAD1.0/test/rm_answers_start.npy')
    sentences = np.load(os.path.join(dataroot, files[0]))['sent']
    questions = np.load(os.path.join(dataroot, files[1]))
    answers = np.load(os.path.join(dataroot, files[2]))
    answers_start = np.load(os.path.join(dataroot, files[3]))
    return sentences, questions, answers, answers_start


def import_entity_data(data_type=0, is_connected=False, is_span_bounds=False):
    if is_connected:
        if data_type == 0:
            entity_file = './processed/SQuAD1.0/train/connected_entity.npy'
            entity_location_file = './processed/SQuAD1.0/train/connected_entity_location.npy'
            entity_span_bounds_file = './processed/SQuAD1.0/train/connected_entity_span_bounds.npy'
            entity_type_file = './processed/SQuAD1.0/train/connected_entity_type.npy'
        elif data_type == 1:
            entity_file = './processed/SQuAD1.0/val/connected_entity.npy'
            entity_location_file = './processed/SQuAD1.0/val/connected_entity_location.npy'
            entity_span_bounds_file = './processed/SQuAD1.0/val/connected_entity_span_bounds.npy'
            entity_type_file = './processed/SQuAD1.0/val/connected_entity_type.npy'
        else:
            entity_file = './processed/SQuAD1.0/test/connected_entity.npy'
            entity_location_file = './processed/SQuAD1.0/test/connected_entity_location.npy'
            entity_span_bounds_file = './processed/SQuAD1.0/test/connected_entity_span_bounds.npy'
            entity_type_file = './processed/SQuAD1.0/test/connected_entity_type.npy'
    else:
        if data_type == 0:
            entity_file = './processed/SQuAD1.0/train/entity.npy'
            entity_location_file = './processed/SQuAD1.0/train/entity_location.npy'
            entity_span_bounds_file = './processed/SQuAD1.0/train/entity_span_bounds.npy'
            entity_type_file = './processed/SQuAD1.0/train/entity_type.npy'
        elif data_type == 1:
            entity_file = './processed/SQuAD1.0/val/entity.npy'
            entity_location_file = './processed/SQuAD1.0/val/entity_location.npy'
            entity_span_bounds_file = './processed/SQuAD1.0/val/entity_span_bounds.npy'
            entity_type_file = './processed/SQuAD1.0/val/entity_type.npy'
        else:
            entity_file = './processed/SQuAD1.0/test/entity.npy'
            entity_location_file = './processed/SQuAD1.0/test/entity_location.npy'
            entity_span_bounds_file = './processed/SQuAD1.0/test/entity_span_bounds.npy'
            entity_type_file = './processed/SQuAD1.0/test/entity_type.npy'

    entities = np.load(os.path.join(dataroot, entity_file))
    entities_location = np.load(os.path.join(dataroot, entity_location_file))
    entities_type = np.load(os.path.join(dataroot, entity_type_file))
    if not is_span_bounds:
        return entities, entities_location, entities_type
    else:
        entities_span_bounds = np.load(os.path.join(dataroot, entity_span_bounds_file))
        return entities, entities_location, entities_span_bounds, entities_type


def save_entity_data(entities, entities_location, entities_type, data_type=0):
    if data_type == 0:
        save_entity_file = './processed/SQuAD1.0/train/entity'
        save_location_file = './processed/SQuAD1.0/train/entity_location'
        save_type_file = './processed/SQuAD1.0/train/entity_type'
    elif data_type == 1:
        save_entity_file = './processed/SQuAD1.0/val/entity'
        save_location_file = './processed/SQuAD1.0/val/entity_location'
        save_type_file = './processed/SQuAD1.0/val/entity_type'
    else:
        save_entity_file = './processed/SQuAD1.0/test/entity'
        save_location_file = './processed/SQuAD1.0/test/entity_location'
        save_type_file = './processed/SQuAD1.0/test/entity_type'

    np.save(os.path.join(dataroot, save_entity_file), entities)
    np.save(os.path.join(dataroot, save_location_file), entities_location)
    np.save(os.path.join(dataroot, save_type_file), entities_type)


def combine_connected_entities(data_type=0):
    sentences, questions, answers_text, _ = importData(data_type=data_type)
    entities, entities_location, entities_type = import_entity_data(data_type=data_type)
    special_types = ['PERSON', 'ORGANIZATION', 'LOCATION', 'MONEY', 'TRANSPORTATION', 'NUMBER', 'HOLIDAY', 'DATE', 'TIME', 'NORP', 'QUANTITY', 'PERCENT', 'EVENT', 'WORK']
    for i, each_sent_entities in enumerate(entities):
        j = 1
        while j < len(each_sent_entities):
            if (entities_type[i][j-1] == special_types[1] and entities_type[i][j] == 'DATE') or (
                    entities_type[i][j-1] == 'DATE' and entities_type[i][j] == special_types[1]):
                j = j + 1
            elif (entities_type[i][j-1] == 'DATE' and entities_type[i][j] == 'PERCENT') or (
                    entities_type[i][j - 1] == 'PERCENT' and entities_type[i][j] == 'DATE'):
                j = j + 1
            elif (entities_type[i][j - 1] == 'PERCENT' and entities_type[i][j] == special_types[1]) or (
                     entities_type[i][j - 1] == special_types[1] and entities_type[i][j] == 'PERCENT'):
                j = j + 1
            elif entities_type[i][j-1] == special_types[1] and entities_type[i][j] == 'QUANTITY':
                j = j + 1
            elif (entities_type[i][j-1] == 'QUANTITY' and entities_type[i][j] == special_types[-2]) or (
                    entities_type[i][j - 1] == special_types[-2] and entities_type[i][j] == 'QUANTITY'):
                j = j + 1
            elif (entities_type[i][j-1] == special_types[-2] and entities_type[i][j] == 'DATE') or (
                    entities_type[i][j - 1] == 'DATE' and entities_type[i][j] == special_types[-2]):
                j = j + 1
            elif entities_type[i][j-1] == 'DATE' and entities_type[i][j] == 'NORP':
                j = j + 1
            elif entities_type[i][j - 1] == 'DATE' and entities_type[i][j] == 'HOLIDAY':
                j = j + 1
            elif entities_type[i][j-1] == 'HOLIDAY' and entities_type[i][j] == 'DATE':
                j = j + 1
            elif entities_type[i][j-1] == 'NORP' and entities_type[i][j] == 'PERCENT':
                j = j + 1
            elif each_sent_entities[j] in each_sent_entities[j-1]:
                each_sent_entities.pop(j)
                entities_location[i].pop(j)
                entities_type[i].pop(j)
            elif (entities_location[i][j-1][1] + 1) >= entities_location[i][j][0]:
                x = each_sent_entities.pop(j-1)
                y = each_sent_entities.pop(j-1)
                combined_start_index = entities_location[i].pop(j-1)[0]
                combined_end_index = entities_location[i].pop(j-1)[1]
                first_type = entities_type[i].pop(j - 1)
                second_type = entities_type[i].pop(j - 1)

                if combined_start_index < combined_end_index:
                    each_sent_entities.insert(j-1, sentences[i][combined_start_index: combined_end_index])
                    entities_location[i].insert(j-1, (combined_start_index, combined_end_index))

                    if first_type == second_type:
                        entities_type[i].insert(j-1, first_type)
                    elif first_type == 'CONCEPT' and second_type in special_types:
                        entities_type[i].insert(j-1, second_type)
                    elif first_type in special_types and second_type == 'CONCEPT':
                        entities_type[i].insert(j-1, first_type)
                    else:
                        if first_type == special_types[0] or second_type == special_types[0]:
                            entities_type[i].insert(j-1, special_types[0])
                        elif first_type == special_types[2] or second_type == special_types[2]:
                            entities_type[i].insert(j-1, special_types[2])
                        elif first_type == special_types[3] or second_type == special_types[3]:
                            entities_type[i].insert(j-1, special_types[3])
                        elif first_type == special_types[4] or second_type == special_types[4]:
                            entities_type[i].insert(j-1, special_types[4])
                        elif (first_type == special_types[1] and second_type == special_types[-1]) or (
                                first_type == special_types[-1] and second_type == special_types[1]):
                            entities_type[i].insert(j-1, special_types[1])
                        # elif (first_type == special_types[1] and second_type == special_types[-2]) or (
                        elif first_type == special_types[-2] and second_type == special_types[1]:
                            entities_type[i].insert(j-1, special_types[-2])
                        elif first_type == special_types[-2] and second_type == special_types[8]:
                            entities_type[i].insert(j-1, special_types[-2])
                        elif (first_type == special_types[1] and second_type == special_types[8]) or (
                                first_type == special_types[8] and second_type == special_types[1]):
                            entities_type[i].insert(j-1, special_types[8])
                        elif (first_type == special_types[5] and second_type in (special_types[1:2]+special_types[6:])) or (
                                first_type in (special_types[1:2]+special_types[6:]) and second_type == special_types[5]):
                            entities_type[i].insert(j-1, special_types[5])
                        elif (first_type == 'DATE' and second_type == 'TIME') or (first_type == 'TIME' and second_type == 'DATE'):
                            entities_type[i].insert(j-1, 'TIME')
                        elif first_type == special_types[-1]:
                            entities_type[i].insert(j-1, second_type)
                        elif second_type == special_types[-1]:
                            entities_type[i].insert(j-1, first_type)
                        elif first_type in (special_types[1:2]+special_types[9:10]) and second_type == special_types[-2]:
                            entities_type[i].insert(j-1, second_type)
                        elif first_type == special_types[-2] and second_type == special_types[9]:
                            entities_type[i].insert(j-1, first_type)
                        elif first_type == 'QUANTITY' and second_type == 'DATE':
                            entities_type[i].insert(j-1, second_type)
                        elif (first_type == 'NORP' and second_type in (special_types[1:2] + special_types[7:8])) or (
                                first_type in (special_types[1:2] + special_types[7:8]) and second_type == 'NORP'):
                            entities_type[i].insert(j-1, 'NORP')
                        elif first_type == special_types[8] and second_type == special_types[6]:
                            entities_type[i].insert(j-1, special_types[8])
                        elif first_type == special_types[-3] and second_type == special_types[9]:
                            entities_type[i].insert(j-1, special_types[9])
                        elif first_type == special_types[9] and second_type == special_types[6]:
                            entities_type[i].insert(j-1, special_types[6])
                        elif first_type == special_types[6] and second_type == special_types[8]:
                            entities_type[i].insert(j-1, special_types[6])
                        elif first_type == special_types[8] and second_type == special_types[-2]:
                            entities_type[i].insert(j-1, special_types[-2])
                        elif first_type == special_types[6] and second_type == special_types[-2]:
                            entities_type[i].insert(j-1, special_types[-2])
                        elif first_type == special_types[-3] and second_type == special_types[8]:
                            entities_type[i].insert(j-1, special_types[-3])
                        elif first_type == special_types[7] and second_type == special_types[10]:
                            entities_type[i].insert(j-1, special_types[7])
                        else:
                            print(sentences[i])
                            print(questions[i])
                            print(answers_text[i])
                            print(x, '\t', y, '\t')
                            print(each_sent_entities[j-1], first_type, second_type)
                            entities_type[i].insert(j - 1, special_types[1])
                            print('#'*80)
                else:
                    j = j - 1
            else:
                 j = j + 1

        if len(each_sent_entities) > 0 and each_sent_entities[-1] == 'rrb':
            each_sent_entities.pop()
            entities_location[i].pop()
            entities_type[i].pop()

        # print(sentences[i])
        # print(entities[i])
        # print(entities_location[i])
        # print('#'*100)

        if i % 1000 == 0:
            print(i)

    if data_type == 0:
        connected_entity_file = './processed/SQuAD1.0/train/connected_entity.npy'
        connected_entity_location_file = './processed/SQuAD1.0/train/connected_entity_location.npy'
        connected_entity_type_file = './processed/SQuAD1.0/train/connected_entity_type.npy'
    elif data_type == 1:
        connected_entity_file = './processed/SQuAD1.0/val/connected_entity.npy'
        connected_entity_location_file = './processed/SQuAD1.0/val/connected_entity_location.npy'
        connected_entity_type_file = './processed/SQuAD1.0/val/connected_entity_type.npy'
    else:
        connected_entity_file = './processed/SQuAD1.0/test/connected_entity.npy'
        connected_entity_location_file = './processed/SQuAD1.0/test/connected_entity_location.npy'
        connected_entity_type_file = './processed/SQuAD1.0/test/connected_entity_type.npy'

    np.save(os.path.join(dataroot, connected_entity_file), entities)
    np.save(os.path.join(dataroot, connected_entity_location_file), entities_location)
    np.save(os.path.join(dataroot, connected_entity_type_file), entities_type)


def remove_unoverlap_data(sentences, questions, answers, answers_start):
    rm_sentences, rm_questions, rm_answers, rm_answers_start = list(), list(), list(), list()

    for i,sent in enumerate(sentences):
        sent_words = nltk.word_tokenize(sent)
        rm_sent_words = [w for w in sent_words if(w not in stopwords.words('english'))]
        rm_sent_words_set = set(rm_sent_words)

        cur_questions, cur_answers, cur_answers_start = list(), list(), list(),
        for j,ques in enumerate(questions[i]):
            ques_words = nltk.word_tokenize(ques)
            rm_ques_words = [w for w in ques_words if(w not in stopwords.words('english'))]
            if len(rm_sent_words_set & set(rm_ques_words)) > 0:
                cur_questions.append(ques)
                cur_answers.append(answers[i][j])
                cur_answers_start.append(answers_start[i][j])
        if len(cur_questions) > 0:
            rm_sentences.append(sent)
            rm_questions.append(cur_questions)
            rm_answers.append(cur_answers)
            rm_answers_start.append(cur_answers_start)
        if i % 100 == 0:
            print(i)
    print('length', len(rm_sentences))
    np.save('../processed/SQuAD1.0/test/rm_sentences', rm_sentences)
    np.save('../processed/SQuAD1.0/test/rm_questions', rm_questions)
    np.save('../processed/SQuAD1.0/test/rm_answers', rm_answers)
    np.save('../processed/SQuAD1.0/test/rm_answers_start', rm_answers_start)


def tag_answers(sentences, answers, answers_start):
    sentences_taggings = list()
    for i, each_sent in enumerate(sentences):
        sent_length = len(nltk.word_tokenize(each_sent))
        for j in range(len(answers[i])):
            BIO_tagging = ['O', ] * sent_length
            start = len(nltk.word_tokenize(each_sent[:answers_start[i][j]]))
            answer_length = len(nltk.word_tokenize(answers[i][j]))
            BIO_tagging[start] = 'B'
            if answer_length > 1:
                BIO_tagging[start+1 : start+answer_length] = ['I', ] * (answer_length - 1)

            sentences_taggings.append(BIO_tagging)
    sentences_taggings = np.array(sentences_taggings)
    return sentences_taggings


def get_tagging_data(answers_BIO, index=None):
    tag_dict = {'B': 1, 'I': 2, 'O': 0}
    all_answers_tag = []
    for answer_bio in answers_BIO:
        vector = list()
        for each_tag in answer_bio:
            idx = tag_dict.get(each_tag)
            vector.append(idx)
        all_answers_tag.append(vector)
    all_answers_tag = np.array(all_answers_tag)
    if index is None:
        shuffle_all_answer_tag = all_answers_tag
    else:
        shuffle_all_answer_tag = all_answers_tag[index]
    return shuffle_all_answer_tag, tag_dict


def save_answer_tagging(data_type=0):
    if data_type == 0:
        answer_taggings_file = './processed/SQuAD1.0/train/answer_labels.npy'
    elif data_type ==1:
        answer_taggings_file = './processed/SQuAD1.0/val/answer_labels.npy'
    else:
        answer_taggings_file = './processed/SQuAD1.0/test/answer_labels.npy'

    sentences, _, answers, answers_start = importData(data_type=data_type)
    answer_taggings = tag_answers(sentences, answers, answers_start)
    all_answer_taggings = get_tagging_data(answer_taggings)[0]
    np.save(os.path.join(dataroot, answer_taggings_file), all_answer_taggings)


if __name__ == '__main__':
    test_sentences, test_questions, test_answers, test_answers_start = importData(data_type=2, is_rm=False)
    remove_unoverlap_data(test_sentences, test_questions, test_answers, test_answers_start)

    # combine_connected_entities(data_type=0)
    # combine_connected_entities(data_type=1)
    # combine_connected_entities(data_type=2)

    save_answer_tagging(data_type=0)
    save_answer_tagging(data_type=1)
    save_answer_tagging(data_type=2)


    pass


