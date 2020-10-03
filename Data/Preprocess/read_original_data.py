import json
import nltk
import numpy as np
start_token = 'sos'
end_token = 'eos'


def extract_qas(type=0):
    train_json, val_json, test_json = '../original/train.json', '../original/dev.json', '../original/test.json'
    if type == 0:
        file = train_json
    elif type == 1:
        file = val_json
    else:
        file = test_json

    qa_dict = dict()
    with open(file, 'r') as f:
        json_data = json.load(f)
        for i in range(len(json_data)):
            each_title_paragraphs = json_data[i]['paragraphs']
            for j in range(len(each_title_paragraphs)):
                each_paragraph = each_title_paragraphs[j]
                each_para_qas = each_paragraph['qas']
                for k in range(len(each_para_qas)):
                    cur_qa = each_para_qas[k]
                    # question = cur_qa['question'].strip()
                    question = cur_qa['question'].strip().lower()
                    question = process_sentence(question)

                    # answer = cur_qa['answers'][0]['text']
                    answer = cur_qa['answers'][0]['text'].lower()
                    answer = process_sentence(answer)
                    qa_dict[question] = answer
        return qa_dict


def process_sentence(question):
    if " '" in question:
        question = question.replace(" '", " ` ")
    if "' " in question:
        question = question.replace("' ", " ' ")
    if "'?" in question:
        question = question.replace("'?", " '?")
    if "'s" in question:
        question = question.replace("'s", " 's")
    if ': ' in question:
        question = question.replace(': ', ' : ')
    if '%' in question:
        question = question.replace('%', ' %')
    if '$' in question:
        question = question.replace('$', '$ ')
    if '=' in question:
        question = question.replace('=', ' = ')
    if '- ' in question and '--' not in question:
        question = question.replace('- ', ' - ')
    if '?' in question and '??' not in question:
        question = question.replace('?', ' ?')
    if '??' in question:
        question = question.replace('??', ' ??')
    if '. ' in question and 'u.s.' not in question and 'jr.' not in question and ' v.' not in question \
            and 'dec.' not in question and 'h.j.' not in question and 'c.w.' not in question \
            and 'st.' not in question and 'dr.' not in question and ' w.' not in question \
            and ' h.' not in question and ' r.' not in question and ' a.' not in question \
            and 'd.c.' not in question and ' d.' not in question and ' mt.' not in question \
            and 'mr.' not in question and 'f.c.' not in question and ' j.' not in question \
            and 'u.e.' not in question and ' c.' not in question and ' m.' not in question \
            and ' b.' not in question and ' no.' not in question and ' op.' not in question \
            and 'rev.' not in question and 'u.k.' not in question and 'l.a.' not in question \
            and 'mrs.' not in question and 'm.sc' not in question and ' f.' not in question \
            and 'inc.' not in question and ' e.' not in question and 'g.m.c.' not in question:
        question = question.replace('. ', ' . ')
    if '(' in question and ')' in question:
        question = question.replace('(', '-lrb- ')
        question = question.replace(')', ' -rrb-')
    if '[' in question and ']' in question:
        question = question.replace('[', '-lsb- ')
        question = question.replace(']', ' -rsb-')
    if '\u27e8' in question and '\u27e9':
        question = question.replace('\u27e8', '\u27e8 ')
        question = question.replace('\u27e9', ' \u27e9')
    if '\u00a3' in question:
        question = question.replace('\u00a3', '# ')
    if '\u2013' in question and ' \u2013 ' not in question:
        question = question.replace('\u2013', ' -- ')
    if "n't" in question:
        question = question.replace("n't", " n't")
    if "'re" in question and " 're" not in question:
        question = question.replace("'re", " 're")
    if "'ve" in question:
        question = question.replace("'ve", " 've")
    while '\"' in question:
        if '\"' in question:
            index = question.find('\"')
            question = question[:index] + '`` ' + question[index + 1:]
        if '\"' in question:
            index = question.find('\"')
            question = question[:index] + " ''" + question[index + 1:]
    if ', ' in question:
        question = question.replace(', ', ' , ')
    if question[-1] == '.' or question[-1] == '>' or question[-1] == '/':
        question = question[:-1] + ' ' + question[-1]
    question = question.replace('   ', ' ')
    question = question.replace('  ', ' ')
    return question


def importData():
    src_file = ('../original/src-train.txt', '../original/src-dev.txt', '../original/src-test.txt')
    tgt_file = ('../original/tgt-train.txt', '../original/tgt-dev.txt', '../original/tgt-test.txt')
    src_train, src_dev, src_test = list(), list(), list()
    tgt_train, tgt_dev, tgt_test = list(), list(), list()
    for i,file in enumerate(src_file):
        with open(file, "r",) as f:
            for line in f.readlines():
                try:
                    content = line.strip()
                    if i == 0:
                        src_train.append(content)
                    elif i == 1:
                        src_dev.append(content)
                    else:
                        src_test.append(content)
                except ValueError:
                    pass
    for i,file in enumerate(tgt_file):
        with open(file, "r",) as f:
            for line in f.readlines():
                try:
                    content = line.strip()
                    content = start_token + ' ' + content + ' ' + end_token
                    if i == 0:
                        tgt_train.append(content)
                    elif i == 1:
                        tgt_dev.append(content)
                    else:
                        tgt_test.append(content)
                except ValueError:
                    pass
    return src_train, src_dev, src_test, tgt_train, tgt_dev, tgt_test


def map_answers(qa_pairs ,type = 0):
    tgt_train_file, tgt_val_file, tgt_test_file = '../original/tgt-train.txt', '../original/tgt-dev.txt', '../original/tgt-test.txt'
    if type == 0:
        tgt_file = tgt_train_file
    elif type == 1:
        tgt_file = tgt_val_file
    else:
        tgt_file = tgt_test_file

    answers = list()
    index_list = list()
    index = 0
    with open(tgt_file, "r", ) as f:
        for line in f.readlines():
            content = line.strip().lower()
            if content in qa_pairs:
                answers.append(qa_pairs[content])
                index_list.append(index)
            index += 1
        return answers, index_list


def match_sentences_qa(src, tgt, answers_list, index_list, limit_length=100):
    sentences, questions, answers, answers_start = list(), list(), list(), list()
    for i,index in enumerate(index_list):
        cur_answer_start = src[index].find(answers_list[i])
        answer_sentence_length = len(nltk.word_tokenize(src[index][:cur_answer_start])) + len(
            nltk.word_tokenize(answers_list[i]))
        if answer_sentence_length <= min(len(nltk.word_tokenize(src[index])), limit_length):
            if src[index] not in sentences:
                sentences.append(src[index])
                questions.append([tgt[index]])
                answers.append([answers_list[i]])
                answers_start.append([cur_answer_start])
            else:
                sentence_position = sentences.index(src[index])
                questions[sentence_position].append(tgt[index])
                answers[sentence_position].append(answers_list[i])
                answers_start[sentence_position].append(cur_answer_start)
    return sentences, questions, answers, answers_start


def calculate_sentences_length():
    sentences = np.load('../processed/SQuAD1.0/train/sentences.npz')['sent']
    num_30, num_50, num_80, num_100, num_more = 0, 0, 0, 0, 0
    for each_sentence in sentences:
        each_length = len(nltk.word_tokenize(each_sentence))
        if each_length <= 30:
            num_30 += 1
        elif each_length <= 50:
            num_50 += 1
        elif each_length <= 80:
            num_80 += 1
        elif each_length <= 100:
            num_100 += 1
        else:
            num_more += 1
    print(num_30, num_50, num_80, num_100, num_more)


def calculate_questions_length():
    questions = np.load('../processed/SQuAD1.0/train/questions.npy')
    num_20, num_30, num_40, num_50, num_more = 0, 0, 0, 0, 0
    for each_question_list in questions:
        for each_question in each_question_list:
            each_length = len(nltk.word_tokenize(each_question))
            if each_length <= 20:
                num_20 += 1
            elif each_length <= 30:
                num_30 += 1
            elif each_length <= 40:
                num_40 += 1
            elif each_length <= 50:
                num_50 += 1
            else:
                num_more += 1
    print(num_20, num_30, num_40, num_50, num_more)


def calculate_distribution():
    questions = np.load('../processed/SQuAD1.0/test/questions.npy')
    num_0 = 0
    num_1, num_2, num_3, num_4, num_5 = 0, 0, 0, 0, 0
    for each_questions in questions:
        if len(each_questions) == 0:
            num_0 += 1
            print(num_0)
        elif len(each_questions) == 1:
            num_1 += 1
        elif len(each_questions) == 2:
            num_2 += 1
        elif len(each_questions) == 3:
            num_3 += 1
        elif len(each_questions) == 4:
            num_4 += 1
        else:
            num_5 += 1
    print(num_1, num_2, num_3, num_4, num_5)


if __name__ == '__main__':
    src_train, src_dev, src_test, tgt_train, tgt_dev, tgt_test = importData()
    print(len(src_train), len(tgt_train), len(src_dev))

    # training data
    qa_dict_train = extract_qas(type=0)
    answers_list_train, index_list_train = map_answers(qa_dict_train, type=0)
    sentences, questions, answers, answers_start = match_sentences_qa(src_train, tgt_train, answers_list_train, index_list_train)
    np.savez_compressed('../processed/SQuAD1.0/train/sentences', sent=sentences)
    np.save('../processed/SQuAD1.0/train/questions', questions)
    np.save('../processed/SQuAD1.0/train/answers', answers)
    np.save('../processed/SQuAD1.0/train/answers_start', answers_start)
    print(sentences[0])
    print(len(sentences), len(answers), len(answers_start), len(questions))
    print(sum(list(map(len, questions))), sum(list(map(len, answers))), sum(list(map(len, answers_start))))
    calculate_sentences_length()
    calculate_questions_length()
    calculate_distribution()

    # validation data
    qa_dict_val = extract_qas(type=1)
    answers_list_val, index_list_val = map_answers(qa_dict_val, type=1)
    sentences, questions, answers, answers_start = match_sentences_qa(src_dev, tgt_dev, answers_list_val, index_list_val)
    np.savez_compressed('../processed/SQuAD1.0/val/sentences', sent=sentences)
    np.save('../processed/SQuAD1.0/val/questions', questions)
    np.save('../processed/SQuAD1.0/val/answers', answers)
    np.save('../processed/SQuAD1.0/val/answers_start', answers_start)
    print(len(sentences), len(answers), len(answers_start), len(questions))
    print(sum(list(map(len, questions))), sum(list(map(len, answers))), sum(list(map(len, answers_start))))
    calculate_distribution()

    # test data
    qa_dict_test = extract_qas(type=2)
    answers_list_test, index_list_test = map_answers(qa_dict_test, type=2)
    sentences, questions, answers, answers_start = match_sentences_qa(src_test, tgt_test, answers_list_test, index_list_test)
    np.savez_compressed('../processed/SQuAD1.0/test/sentences', sent=sentences)
    np.save('../processed/SQuAD1.0/test/questions', questions)
    np.save('../processed/SQuAD1.0/test/answers', answers)
    np.save('../processed/SQuAD1.0/test/answers_start', answers_start)
    print(len(sentences), len(questions), len(answers), len(answers_start))
    print(sum(list(map(len, questions))), sum(list(map(len, answers))), sum(list(map(len, answers_start))))
    calculate_distribution()







