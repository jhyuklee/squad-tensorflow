import sys
import json
import re
import gensim
import datetime
import string
import operator
import numpy as np


def read_data(dataset_path, version):
    with open(dataset_path) as dataset_file:
        dataset_json = json.load(dataset_file)
        if (dataset_json['version'] != version):
            print('Evaluation expects v-' + version +
                  ', but got dataset with v-' + dataset_json['version'],
                  file=sys.stderr)
        dataset = dataset_json['data']
    return dataset


def load_glove(glove_path, dictionary):
    print('Glove Loading...')
    start_time = datetime.datetime.now()
    glove = gensim.models.Word2Vec.load_word2vec_format(glove_path, binary=False)
    elapsed_time = datetime.datetime.now() - start_time
    print('Glove Loading Done', elapsed_time)

    pretrained_vectors = None
    unk_cnt = 0
    for word, vector in sorted(dictionary.items(), key=operator.itemgetter(1)):
        if word in glove:
            word_vector = glove[word]
        else:
            word_vector = np.random.rand(300)
            unk_cnt += 1

        if pretrained_vectors is None:
            pretrained_vectors = [word_vector]
        else:
            pretrained_vectors = np.concatenate((pretrained_vectors, [word_vector]), axis=0)

    print('Pretrained vectors', pretrained_vectors.shape, 'unknown', unk_cnt)
    print('Pretrained sample', pretrained_vectors[dictionary['UNK']])
    print('Pretrained sample', pretrained_vectors[dictionary['good']])
    return pretrained_vectors


def tokenize(words):
    exclude = set(string.punctuation)
    words = ''.join(ch for ch in words if ch not in exclude)
    while '  ' in words:
        words = re.sub(r'\s\s', ' ', words)
    return words.lower().split(' ')


def word2idx(words, dictionary, max_length=None):
    result_idx = []
    for word in tokenize(words):
        if word not in dictionary:
            result_idx.append(dictionary['UNK'])
        else:
            result_idx.append(dictionary[word])

    original_len = len(result_idx)
    if max_length is not None:
        while len(result_idx) < max_length:
            result_idx.append(dictionary['PAD'])

    return result_idx, original_len


def word2cnt(words, counter):
    for word in words:
        if word not in counter:
            counter[word] = 1
        else:
            counter[word] += 1


def build_dictionary(dataset, params):
    dictionary = {}
    reverse_dictionary = {}
    counter = {}
    context_maxlen = 0
    question_maxlen = 0
    answer_maxlen = 0
    dictionary['UNK'] = 0
    dictionary['PAD'] = 1
    reverse_dictionary[0] = 'UNK'
    reverse_dictionary[1] = 'PAD'
    
    for d_idx, document in enumerate(dataset):
        for p_idx, paragraph in enumerate(document['paragraphs']):
            context = paragraph['context']
            context_words = tokenize(context)
            word2cnt(context_words, counter)
            if len(context_words) > context_maxlen:
                context_maxlen = len(context_words)
            
            for qa in paragraph['qas']:
                question = qa['question']
                answers = qa['answers']
                question_words = tokenize(question)
                word2cnt(question_words, counter)
                if len(question_words) > question_maxlen:
                    question_maxlen = len(question_words)

                for answer in answers:
                    answer_words = tokenize(answer['text'])
                    word2cnt(answer_words, counter)
                    if len(answer_words) > answer_maxlen:
                        answer_maxlen = len(answer_words)
    
    print('Top 20 frequent words among', len(counter))
    print([(k, counter[k]) for k in sorted(counter, key=counter.get, reverse=True)[:20]])
    for key, value in counter.items():
        if value > params['min_voca']:
            dictionary[key] = len(dictionary)
            reverse_dictionary[dictionary[key]] = key
    print('Dictionary size', len(dictionary))
    print([(k, dictionary[k]) for k in sorted(dictionary, key=dictionary.get)[:20]])
    print('Maxlen of C:%d, Q:%d, A:%d' % (context_maxlen, question_maxlen, answer_maxlen))

    return dictionary, reverse_dictionary, context_maxlen, question_maxlen


def preprocess(dataset, dictionary, c_maxlen, q_maxlen):
    cqa_set = [] 

    for d_idx, document in enumerate(dataset):
        for p_idx, paragraph in enumerate(document['paragraphs']):
            context = paragraph['context']
            cqa_item = {}
            cqa_item['c_raw'] = tokenize(context)
            cqa_item['c'], cqa_item['c_len'] = word2idx(context, dictionary, c_maxlen)
            if d_idx == 0 and p_idx == 0:
                # print(context)
                pass
            qa_set = []
            for qa in paragraph['qas']:
                qa_item = {}
                question = qa['question']
                answers = qa['answers']
                qa_item['q'], qa_item['q_len'] = word2idx(question, dictionary, q_maxlen)
                qa_item['a_start'] = len(tokenize(context[:answers[0]['answer_start']]))
                qa_item['a_end'] = qa_item['a_start'] + len(tokenize(answers[0]['text'])) - 1
                qa_item['a'] = [a['text'] for a in answers]
                qa_set.append(qa_item)
                if d_idx == 0 and p_idx == 0:
                    # print(question)
                    # print(answers)
                    pass
            cqa_item['qa'] = qa_set
            cqa_set.append(cqa_item)

    # print('\nis preprocessed as \n')
    # print(cqa_set[0])

    return cqa_set

