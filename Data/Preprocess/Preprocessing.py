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

    save_answer_tagging(data_type=0)
    save_answer_tagging(data_type=1)
    save_answer_tagging(data_type=2)


    pass


