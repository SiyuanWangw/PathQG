import numpy as np
start_token = 'sos'
end_token = 'eos'


def indexs2str(indexs, vocabulary, is_question=False):
    string = ''
    for i in indexs:
        if is_question is True and vocabulary[i] == end_token:
            break
        if vocabulary[i] != 'UNK':
            string += vocabulary[i] + ' '

    return string


def indexs2str_UNK(indexs, vocabulary, vocabulary_src, cur_input_encode, alpha, is_question=True):
    string = ''
    for n,i in enumerate(indexs):
        if vocabulary[i] == 'UNK':
            input_replace_index = np.argmax(alpha[n])
            replace_word = vocabulary_src[cur_input_encode[input_replace_index]]
            string += replace_word
        else:
            if is_question is True and vocabulary[i] == end_token:
                break
            string += vocabulary[i] + ' '
    if is_question:
        string = start_token + ' ' + string + ' ' + end_token
    return string


def print_answers(start, end, sent_index, vocabulary):
    answer_index = sent_index[start: end+1]
    print(start, end, indexs2str(answer_index, vocabulary))


def predict_answers(prediction, num):
    final_predict_start, final_predict_end = list(), list()
    for i in range(num):
        index_start = np.argmax(prediction[0])
        index_end = np.argmax(prediction[1])
        final_predict_start.append(index_start)
        final_predict_end.append(index_end)
        prediction[0][index_start] = 0
        prediction[1][index_end] = 0
    return final_predict_start, final_predict_end


def convert_tagging(predict_start, predict_end, tagging_length):
    BIO_tagging = [0, ] * tagging_length
    extract_num = 0
    for j in range(len(predict_start)):
        begin_index = predict_start[j]
        end_index = predict_end[j]
        answer_length = end_index - begin_index + 1
        if answer_length > 0:
            BIO_tagging[begin_index] = 1
            extract_num += 1
        if answer_length > 1:
            BIO_tagging[begin_index + 1: begin_index + answer_length] = [2, ] * (answer_length - 1)
    if extract_num > 0:
        return BIO_tagging, True, extract_num
    else:
        return BIO_tagging, False, extract_num


def convert_single_tagging(predict_start, predict_end, tagging_length):
    extract_num = 0
    BIO_tagging = [0, ] * tagging_length
    begin_index = predict_start
    end_index = predict_end
    answer_length = end_index - begin_index + 1
    if answer_length > 0:
        BIO_tagging[begin_index] = 1
        extract_num += 1
        if answer_length > 1:
            BIO_tagging[begin_index + 1: begin_index + answer_length] = [2, ] * (answer_length - 1)

    return BIO_tagging, extract_num


def compute_Match(predict, label):
    correct_num = 0.0
    # correct_prediction = list()
    for i in predict:
        if i in label:
            correct_num += 1
            # correct_prediction.append(i)
    correct_num = correct_num/len(label)
    return correct_num


def compute_ExactMatch(predict_start, label_start, predict_end, label_end):
    correct_num = 0.0
    for i in range(len(predict_start)):
        if predict_start[i] in label_start and predict_end[i] in label_end:
            correct_num += 1
    correct_num = correct_num / len(label_start)
    return correct_num


def calculate_F1(label, prediction):
    TP, FP, FN = 0, 0, 0
    for i in range(len(label)):
        if label[i] != 0 and prediction[i] != 0:
            TP += 1
        elif label[i] != 0:
            FN += 1
        elif prediction[i] != 0:
            FP += 1
    if TP == 0:
        return 0, False
    else:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score = 2*precision*recall/(precision+recall)
        return f1_score, True