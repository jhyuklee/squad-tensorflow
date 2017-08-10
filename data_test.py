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
from utils import process_tokens
from pycorenlp import StanfordCoreNLP 

nlp = StanfordCoreNLP('http://localhost:9000')
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
    np.random.seed(253)
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


def tokenize_corenlp(words):
    output = nlp.annotate(
            words,properties = {'annotators':'tokenize','outputFormat':'json'})
    result = [w['word'].replace("''",'"').replace("``",'"') for w in output['tokens']]
    return result    



def tokenize(words):
    result = [token.replace("''", '"').replace("``", '"') 
            for token in nltk.word_tokenize(words)]
    result = [process_tokens(tokens) for tokens in [result]]  # process tokens
    return result[0]

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

def char2idx(words, char_dictionary, word_maxlen = None, max_length = None):
    result_idx = []
    word_idx = []
    word_length = []
    for word in words:
        for char in word:
            if char not in char_dictionary:
                word_idx.append(char_dictionary['UNK'])
            else:
                word_idx.append(char_dictionary[char])
        
        word_length.append(len(word_idx))
        if word_maxlen is not None:
            while len(word_idx) < word_maxlen:
                word_idx.append(char_dictionary['PAD'])
            if len(word_idx) > word_maxlen:
                word_idx = word_idx[:word_maxlen]
        result_idx.append(word_idx)
        word_idx = []
    
    if max_length is not None:
        while len(result_idx) < max_length:
            result_idx.append([char_dictionary['PAD']]*word_maxlen)
            word_length.append(0)
        if len(result_idx) > max_length:
            result_idx = result_idx[:max_length]
            word_length = word_length[:max_length]
    
    return result_idx, word_length 


def word2cnt(words, counter):
    for word in words:
        if word not in counter:
            counter[word] = 1
        else:
            counter[word] += 1

def char2cnt(word_list, counter):
    for word in word_list:
        for char in word:
            if char not in counter:
                counter[char] = 1
            else:
                counter[char] += 1


def build_dict(dataset, params):
    dictionary = {}
    reverse_dictionary = {}
    counter = {}
    context_maxlen = 0
    question_maxlen = 0
    answer_maxlen = 0
    
    char_dict = {}
    reverse_char_dict = {}
    word_maxlen = 0
    char_counter = {}
    char_dict['UNK'] = 0
    char_dict['PAD'] = 1
    reverse_char_dict[0] = 'UNK'
    reverse_char_dict[1] = 'PAD'
    
    for d_idx, document in enumerate(dataset):
        for p_idx, paragraph in enumerate(document['paragraphs']):
            context = paragraph['context']
            context_words = tokenize(context)
            context_seo = context_split_seo(context)
            if context_words != context_seo:
                print("%%% not match !!!")
                print("our : ", context_words, len(context_words))
                print("seo : ", context_seo, len(context_seo))
            c_char = [list(word) for word in context_words]
	    
            word2cnt(context_words, counter)
            char2cnt(c_char, char_counter)

            if len(context_words) > context_maxlen:
                context_maxlen = len(context_words)
	    
            for word in c_char:
                if len(word) > word_maxlen:
                    word_maxlen = len(word)


            for qa in paragraph['qas']:
                question = qa['question']
                answers = qa['answers']
                question_words = tokenize(question)
                q_char = [list(word) for word in question_words]
                
                word2cnt(question_words, counter)
                char2cnt(q_char, char_counter)

                if len(question_words) > question_maxlen:
                    question_maxlen = len(question_words)

                for answer in answers:
                    answer_words = tokenize(answer['text'])
                    word2cnt(answer_words, counter)
                    if len(answer_words) > answer_maxlen:
                        answer_maxlen = len(answer_words)
		
                for word in q_char:
                    if len(word) > word_maxlen:
                        word_maxlen = len(word)

    print('Top 20 frequent words among', len(counter))
    print([(k, counter[k]) for k in sorted(
        counter, key=counter.get, reverse=True)[:20]])
    tester = sorted(counter, key=counter.get, reverse=True)[70000]
    print(tester, counter[tester])
    
    print('Top 20 frequent characters among', len(char_counter))
    print([(k,char_counter[k]) for k in sorted(
        char_counter, key=char_counter.get, reverse=True)[:20]])

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
    
    for key, value in sorted(char_counter.items()):
        char_dict[key] = len(char_dict)
        reverse_char_dict[char_dict[key]] = key

    print('Dictionary size', len(dictionary))
    print([(k, dictionary[k]) for k in sorted(dictionary, key=dictionary.get)[:20]])
    
    print('Charcter dict size',len(char_dict))
    print([(k,char_dict[k]) for k in sorted(char_dict, key = char_dict.get)[:20]])
    
    print('Maxlen of C:%d, Q:%d, A:%d' % (
        context_maxlen, question_maxlen, answer_maxlen))
    print('Word maxlen : %d' % word_maxlen)
   
    return (dictionary, reverse_dictionary, context_maxlen, 
            question_maxlen, word_maxlen, char_dict, reverse_char_dict)




def get_word2vec(args, word_counter):
    glove_path = os.path.join(args.glove_dir, "glove.{}.{}d.txt".format(args.glove_corpus, args.glove_vec_size))
    sizes = {'6B': int(4e5), '42B': int(1.9e6), '840B': int(2.2e6), '2B': int(1.2e6)}
    total = sizes[args.glove_corpus]
    word2vec_dict = {}
    with open(glove_path, 'r', encoding='utf-8') as fh:
        for line in tqdm(fh, total=total):
            array = line.lstrip().rstrip().split(" ")
            word = array[0]
            vector = list(map(float, array[1:]))
            if word in word_counter:
                word2vec_dict[word] = vector
            elif word.capitalize() in word_counter:
                word2vec_dict[word.capitalize()] = vector
            elif word.lower() in word_counter:
                word2vec_dict[word.lower()] = vector
            elif word.upper() in word_counter:
                word2vec_dict[word.upper()] = vector
    json.dump(word2vec_dict, open('word_dict.json','a'))
    print("{}/{} of word vocab have corresponding vectors in {}".format(len(word2vec_dict), len(word_counter), glove_path))
    return word2vec_dict



def context_split_seo(context):
    sent_tokenize = nltk.sent_tokenize
    def word_tokenize(tokens):
        return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]
    # wordss
    context = context.replace("''", '" ')
    context = context.replace("``", '" ')
    xi = list(map(word_tokenize, sent_tokenize(context)))
    xi = [process_tokens(tokens) for tokens in xi]  # process tokens
    res = []
    for s in xi:
        for w in s:
            res.append(w)
    # given xi, add chars
    cxi = [[list(xijk) for xijk in xij] for xij in xi]
    return res


def prepro_each(args, data_type, start_ratio=0.0, stop_ratio=1.0, out_name="default", in_path=None):
    if args.tokenizer == "PTB":
        import nltk
        sent_tokenize = nltk.sent_tokenize
        def word_tokenize(tokens):
            return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]
    elif args.tokenizer == 'Stanford':
        from my.corenlp_interface import CoreNLPInterface
        interface = CoreNLPInterface(args.url, args.port)
        sent_tokenize = interface.split_doc
        word_tokenize = interface.split_sent
    else:
        raise Exception()

    if not args.split:
        sent_tokenize = lambda para: [para]

    source_path = in_path or os.path.join(args.source_dir, "{}-{}v1.1.json".format(data_type, args.suffix))
    source_data = json.load(open(source_path, 'r'))

    q, cq, y, rx, rcx, ids, idxs = [], [], [], [], [], [], []
    na = []
    cy = []
    x, cx = [], []
    answerss = []
    p = []
    word_counter, char_counter, lower_word_counter = Counter(), Counter(), Counter()
    start_ai = int(round(len(source_data['data']) * start_ratio))
    stop_ai = int(round(len(source_data['data']) * stop_ratio))
    for ai, article in enumerate(tqdm(source_data['data'][start_ai:stop_ai])):
        xp, cxp = [], []
        pp = []
        x.append(xp)
        cx.append(cxp)
        p.append(pp)
        for pi, para in enumerate(article['paragraphs']):
            # wordss
            context = para['context']
            context = context.replace("''", '" ')
            context = context.replace("``", '" ')
            xi = list(map(word_tokenize, sent_tokenize(context)))
            xi = [process_tokens(tokens) for tokens in xi]  # process tokens
            # given xi, add chars
            cxi = [[list(xijk) for xijk in xij] for xij in xi]
            xp.append(xi)
            cxp.append(cxi)
            pp.append(context)

            for xij in xi:
                for xijk in xij:
                    word_counter[xijk] += len(para['qas'])
                    lower_word_counter[xijk.lower()] += len(para['qas'])
                    for xijkl in xijk:
                        char_counter[xijkl] += len(para['qas'])

            rxi = [ai, pi]
            assert len(x) - 1 == ai
            assert len(x[ai]) - 1 == pi
            for qa in para['qas']:
                # get words
                qi = word_tokenize(qa['question'])
                qi = process_tokens(qi)
                cqi = [list(qij) for qij in qi]
                yi = []
                cyi = []
                answers = []
                for answer in qa['answers']:
                    answer_text = answer['text']
                    answers.append(answer_text)
                    answer_start = answer['answer_start']
                    answer_stop = answer_start + len(answer_text)
                    # TODO : put some function that gives word_start, word_stop here
                    yi0, yi1 = get_word_span(context, xi, answer_start, answer_stop)
                    # yi0 = answer['answer_word_start'] or [0, 0]
                    # yi1 = answer['answer_word_stop'] or [0, 1]
                    assert len(xi[yi0[0]]) > yi0[1]
                    assert len(xi[yi1[0]]) >= yi1[1]
                    w0 = xi[yi0[0]][yi0[1]]
                    w1 = xi[yi1[0]][yi1[1]-1]
                    i0 = get_word_idx(context, xi, yi0)
                    i1 = get_word_idx(context, xi, (yi1[0], yi1[1]-1))
                    cyi0 = answer_start - i0
                    cyi1 = answer_stop - i1 - 1
                    # print(answer_text, w0[cyi0:], w1[:cyi1+1])
                    assert answer_text[0] == w0[cyi0], (answer_text, w0, cyi0)
                    assert answer_text[-1] == w1[cyi1]
                    assert cyi0 < 32, (answer_text, w0)
                    assert cyi1 < 32, (answer_text, w1)

                    yi.append([yi0, yi1])
                    cyi.append([cyi0, cyi1])

                if len(qa['answers']) == 0:
                    yi.append([(0, 0), (0, 1)])
                    cyi.append([0, 1])
                    na.append(True)
                else:
                    na.append(False)

                for qij in qi:
                    word_counter[qij] += 1
                    lower_word_counter[qij.lower()] += 1
                    for qijk in qij:
                        char_counter[qijk] += 1

                q.append(qi)
                cq.append(cqi)
                y.append(yi)
                cy.append(cyi)
                rx.append(rxi)
                rcx.append(rxi)
                ids.append(qa['id'])
                idxs.append(len(idxs))
                answerss.append(answers)

        if args.debug:
            break

    word2vec_dict = get_word2vec(args, word_counter)
    lower_word2vec_dict = get_word2vec(args, lower_word_counter)

    # add context here
    data = {'q': q, 'cq': cq, 'y': y, '*x': rx, '*cx': rcx, 'cy': cy,
            'idxs': idxs, 'ids': ids, 'answerss': answerss, '*p': rx, 'na': na}
    shared = {'x': x, 'cx': cx, 'p': p,
              'word_counter': word_counter, 'char_counter': char_counter, 'lower_word_counter': lower_word_counter,
              'word2vec': word2vec_dict, 'lower_word2vec': lower_word2vec_dict}

    print("saving ...")
    save(args, data, shared, out_name)

