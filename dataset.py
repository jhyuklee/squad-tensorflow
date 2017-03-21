import sys
import json
import gensim
import datetime


def read_data(dataset_path, version):
    with open(dataset_path) as dataset_file:
        dataset_json = json.load(dataset_file)
        if (dataset_json['version'] != version):
            print('Evaluation expects v-' + version +
                  ', but got dataset with v-' + dataset_json['version'],
                  file=sys.stderr)
        dataset = dataset_json['data']
    return dataset


def load_glove(glove_path):
    print('Glove Loading...')
    start_time = datetime.datetime.now()
    glove = gensim.models.Word2Vec.load_word2vec_format(glove_path, binary=False)
    elapsed_time = datetime.datetime.now() - start_time
    print('Glove Loading Done', elapsed_time)
    return glove


def tokenize(words):
    # TODO: Normalize text 
    return words.split(' ')


def word2idx(words, dictionary):
    result_idx = []
    for word in tokenize(words):
        if word not in dictionary:
            result_idx.append('UNK')
        else:
            result_idx.append(dictionary[word])
    return result_idx


def word2cnt(words, counter):
    for word in tokenize(words):
        if word not in counter:
            counter[word] = 1
        else:
            counter[word] += 1


def build_dictionary(dataset, params):
    dictionary = {}
    counter = {}
    
    for d_idx, document in enumerate(dataset):
        for p_idx, paragraph in enumerate(document['paragraphs']):
            context = paragraph['context']
            word2cnt(context.lower(), counter)
            for qa in paragraph['qas']:
                question = qa['question']
                answers = qa['answers']
                word2cnt(question.lower(), counter)
                for answer in answers:
                    word2cnt(answer['text'].lower(), counter)
    
    print('Top 20 frequent words among', len(counter))
    print([(k, counter[k]) for k in sorted(counter, key=counter.get, reverse=True)[:20]])
    for key, value in counter.items():
        if value > params['min_voca']:
            dictionary[key] = len(dictionary)
    print('Dictionary size', len(dictionary))

    return dictionary


def preprocess(dataset, dictionary):
    cqa_set = [] 
    print()

    for d_idx, document in enumerate(dataset):
        for p_idx, paragraph in enumerate(document['paragraphs']):
            context = paragraph['context']
            cqa_item = {}
            cqa_item['c'] = word2idx(context, dictionary)
            if d_idx == 0 and p_idx == 0:
                # print(context)
                pass
            qa_set = []
            for qa in paragraph['qas']:
                qa_item = {}
                question = qa['question']
                answers = qa['answers']
                qa_item['q'] = word2idx(question, dictionary)
                qa_item['a'] = [word2idx(answer['text'], dictionary) for answer in answers]
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

