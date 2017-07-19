import sys
import os
import json
import re
import datetime
import string
import operator
import collections
import numpy as np
import nltk
nltk.download('punkt')


def read_data(dataset_path, version):
    with open(dataset_path) as dataset_file:
        dataset_json = json.load(dataset_file)
        if (dataset_json['version'] != version):
            print('Evaluation expects v-' + version +
                  ', but got dataset with v-' + dataset_json['version'],
                  file=sys.stderr)
        dataset = dataset_json['data']
    return dataset


def load_glove(dictionary, params):
    print('Glove Loading...')
    start_time = datetime.datetime.now()
    glove = {}
    glove_path = os.path.expanduser(params['glove_path'])
    with open(glove_path, 'r', encoding='utf-8', errors='ignore') as f:
        while True:
            try:
                line = f.readline()
                if not line: break
                word = line.split()[0]
                embed = [float(l) for l in line.split()[1:]]
                glove[word] = embed
            except ValueError as e:
                print(e)
                
    elapsed_time = datetime.datetime.now() - start_time
    print('Glove Loading Done', elapsed_time, len(glove))

    pretrained_vectors = []
    word2idx = {}
    idx2word = {}
    unk_cnt = 0
    unknown_vector = np.random.uniform(-1, 1, params['dim_embed_word'])
    word2idx['UNK'] = len(word2idx)
    word2idx['PAD'] = len(word2idx)
    idx2word[0] = 'UNK'
    idx2word[1] = 'PAD'
    pretrained_vectors.append(unknown_vector)
    pretrained_vectors.append([0.0] * params['dim_embed_word'])
    for word, word_idx in sorted(dictionary.items(), key=operator.itemgetter(1)):
        if word in glove:
            word2idx[word] = len(word2idx)
            idx2word[len(word2idx)-1] = word
            pretrained_vectors.append(glove[word])
        else:
            unk_cnt += 1

    print('apple:', word2idx['apple'], glove['apple'][:5])
    print('Pretrained vectors', np.asarray(pretrained_vectors).shape, 'unk', unk_cnt)
    print('Dictionary Change', len(dictionary), 'to', len(word2idx), len(idx2word))
    return np.asarray(pretrained_vectors).astype(np.float32), word2idx, idx2word 


def tokenize(words):
    result = [token.replace("''", '"').replace("``", '"').lower() 
            for token in nltk.word_tokenize(words)]
    return result

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
        if len(result_idx) > max_length:
            result_idx = result_idx[:max_length]

    return result_idx, original_len


def word2cnt(words, counter):
    for word in words:
        if word not in counter:
            counter[word] = 1
        else:
            counter[word] += 1


def build_dict(dataset, params):
    dictionary = {}
    reverse_dictionary = {}
    counter = {}
    context_maxlen = 0
    question_maxlen = 0
    answer_maxlen = 0
    
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
    tester = sorted(counter, key=counter.get, reverse=True)[70000]
    print(tester, counter[tester])

    digit = 0
    non_alnum = 0
    for key, value in sorted(counter.items()):
        if key.isdigit(): # TODO: set to DIGIT
            digit += 1
        if not key.isdigit() and not key.isalnum():
            non_alnum += 1

        dictionary[key] = len(dictionary)
        reverse_dictionary[dictionary[key]] = key
    print('digit cnt', digit)
    print('non alpha cnt', non_alnum)

    print('Dictionary size', len(dictionary))
    print([(k, dictionary[k]) for k in sorted(dictionary, key=dictionary.get)[:20]])
    print('Maxlen of C:%d, Q:%d, A:%d' % (context_maxlen, question_maxlen, answer_maxlen))

    return dictionary, reverse_dictionary, context_maxlen, question_maxlen


def preprocess(dataset, dictionary, c_maxlen, q_maxlen):
    cqa_set = []
    cnt = 0

    for d_idx, document in enumerate(dataset):
        for p_idx, paragraph in enumerate(document['paragraphs']):
            context = paragraph['context']
            cqa_item = {}
            cqa_item['c_raw'] = tokenize(context)
            cqa_item['c_real'] = context
            cqa_item['c'], cqa_item['c_len'] = word2idx(context, dictionary, c_maxlen)
            if len(cqa_item['c_raw']) > c_maxlen: continue
            if d_idx == 0 and p_idx == 0:
                # print(context)
                pass
            qa_set = []
            for qa in paragraph['qas']:
                cnt += 1
                qa_item = {}
                question = qa['question']
                answers = qa['answers']
                qa_item['q_raw'] = tokenize(question)
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
    print('Passage: %d, Question: %d' % (len(cqa_set), cnt))

    return cqa_set

