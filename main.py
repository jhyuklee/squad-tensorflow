import sys
import tensorflow as tf
import numpy as np
import pprint
import argparse
import datetime
import random
import copy
import os

from model import Basic
from mpcm import MPCM
from ql_mpcm import QL_MPCM
from bidaf import BiDAF
from my_bidaf import My_BiDAF
from time import gmtime, strftime
from dataset import read_data, build_dict, load_glove, preprocess, load_lm
from run import run_epoch

flags = tf.app.flags
# Basic model settings
flags.DEFINE_integer("batch_size", 32, "Size of batch (32)")
flags.DEFINE_integer("dim_embed_word", 300, "Dimension of word embedding (300)")
flags.DEFINE_integer("dim_rnn_cell", 100, "Dimension of RNN cell (100)")
flags.DEFINE_integer("dim_hidden", 100, "Dimension of hidden layer")
flags.DEFINE_integer("rnn_layer", 1, "Layer number of RNN ")
flags.DEFINE_integer("context_maxlen", 0, "Predefined context length (0 for max)")
flags.DEFINE_float("rnn_dropout", 0.5, "Dropout of RNN cell")
flags.DEFINE_float("hidden_dropout", 0.5, "Dropout rate of hidden layer")
flags.DEFINE_float("embed_dropout", 0.8, "Dropout rate of embedding layer")
flags.DEFINE_float("learning_rate", 0.00162, "Init learning rate of the optimzier")
flags.DEFINE_float("max_grad_norm", 5.0, "Maximum gradient to clip")
flags.DEFINE_string("optimizer", "a", "[s]sgd [m]momentum [a]adam")

# Run options
flags.DEFINE_integer('train_epoch', 100, 'Training epoch')
flags.DEFINE_integer('test_epoch', 1, 'Test for every n training epoch')
flags.DEFINE_integer("validation_cnt", 100, "Number of model validation")
flags.DEFINE_boolean("debug", False, "True to show debug message")
flags.DEFINE_boolean("save", False, "True to save model after testing")
flags.DEFINE_boolean("sample_params", False, "True to sample parameters")
flags.DEFINE_boolean("early_stop", False, "True to make early stop")
flags.DEFINE_boolean("load", False, "True to load model")
flags.DEFINE_boolean("train", True, "True to train model")
flags.DEFINE_boolean("summarize", False, "True to have summarization")
flags.DEFINE_boolean("embed_trainable", False, "True to optimize embedded words")
flags.DEFINE_string("load_name", "m100_300d6B", "load model name")
flags.DEFINE_string("model_name", "none", "Replaced by load_name or auto-named")
flags.DEFINE_string("mode", "m", "b: basic, m: mpcm, q: ql_mpcm")
flags.DEFINE_string("ymdhms", "none", "Model index (ymdhMs)")

# MPCM settings
flags.DEFINE_integer("dim_perspective", 20, "Maximum number of perspective (20)")

# Paraphrase settings
flags.DEFINE_integer("num_paraphrase", 1, "Maximum iter of question paraphrasing")
flags.DEFINE_integer("dim_action", 5, "Dimension of action space")
flags.DEFINE_integer("max_action", 0, "Maximum possible sampled actions")
flags.DEFINE_integer("rb_clip", 2, "Maximum R/B clip")
flags.DEFINE_integer("pp_dim_rnn_cell", 100, "Dimension of RNN cell (100)")
flags.DEFINE_integer("pp_rnn_layer", 1, "Layer number of RNN")
flags.DEFINE_string("policy_q", "e", "question [e] embed [h] hidden")
flags.DEFINE_string("policy_c", "e", "context [e] embed [h] hidden")
flags.DEFINE_string("similarity_q", "e", "question [e] embed [h] hidden")
flags.DEFINE_string("similarity_c", "e", "context [e] embed [h] hidden")
flags.DEFINE_float("reg_param", 0.0, "Regularization parameter")
flags.DEFINE_float("init_exp", 0.0, "Initial exploration prob")
flags.DEFINE_float("final_exp", 0.0, "Final exploration prob")
flags.DEFINE_boolean("anneal_exp", False, "True to anneal exploration")
flags.DEFINE_boolean("train_pp_only", True, "True to train paraphrase only")

# Bidaf settings
flags.DEFINE_integer("highway_num_layers", 2, "highway_num_layers [2]")
flags.DEFINE_integer("hidden_size", 100, "Hidden size [100]")
flags.DEFINE_float("input_keep_prob", 0.8, "Input keep prob of LSTM weights [0.8]")
flags.DEFINE_float("wd", 0.0, "L2 weight decay for regularization [0.0]")
flags.DEFINE_boolean("share_lstm_weights", True, "Share LSTM weights [True]")
flags.DEFINE_boolean("use_char_emb", True, "use char emb? [True]")
flags.DEFINE_boolean("use_word_emb", True, "use word emb? [True]")
flags.DEFINE_boolean("highway", True, "Use highway? [True]")
flags.DEFINE_boolean('load_seo', True, "load Seo's pretrained bidaf")
flags.DEFINE_string('logit_func', 'tri_linear', 'logit func [tri_linear]')
flags.DEFINE_string('answer_func', 'linear', 'answer logit func [linear]')

# Path settings
flags.DEFINE_string('checkpoint_dir', './results/ckpt/', 'Checkpoint directory')
flags.DEFINE_string('summary_dir', './results/summary/', 'summary writer')
flags.DEFINE_string('train_path', './data/train-v1.1.json', 'Training dataset path')
flags.DEFINE_string('dev_path', './data/dev-v1.1.json',  'Development dataset path')
flags.DEFINE_string('pred_path', './results/dev-v1.1-pred.json', 'Pred output path')
flags.DEFINE_string('lm_path', './data/langmodel.pkl', 'Pretrained LM path')
flags.DEFINE_string("glove_size", "6", "use 6B or 840B for glove")
flags.DEFINE_string('glove_path', \
        ('~/common/glove/glove.'+ tf.app.flags.FLAGS.glove_size + 'B.'
            + str(tf.app.flags.FLAGS.dim_embed_word) +'d.txt'), 'embed path')
flags.DEFINE_string('validation_path', './results/validation.txt', 'Validation path')

# Character embedding
flags.DEFINE_string('char_emb_dim', 8,'Character embedding dimension')
flags.DEFINE_string('filter_width', 5, 'CNN fiter width')
flags.DEFINE_string('cnn_layer',1, 'Number of CNN layer')
flags.DEFINE_string('char_out',100,'Character output dim (num of filter)') # TODO
flags.DEFINE_string('share_conv',True,'Share cnn for context and question')
flags.DEFINE_string('cnn_keep_prob',0.8,'Dropout for CNN layer')


FLAGS = flags.FLAGS


def run(model, params, train_dataset, dev_dataset, idx2word):
    max_em = max_f1 = max_ep = es_cnt = 0
    train_epoch = params['train_epoch']
    test_epoch = params['test_epoch']
    init_lr = params['learning_rate']
    early_stop = params['early_stop']
    train_iter = valid_iter = 0
    if params['mode'] == 'q':
        LM = load_lm(params['lm_path']) 
    else:
        LM = None

    for epoch_idx in range(train_epoch):
        if params['train']:
            start_time = datetime.datetime.now()
            print("\n[Epoch %d]" % (epoch_idx + 1))
            _, _, _, train_iter = run_epoch(model, train_dataset, epoch_idx + 1, 
                    train_iter, idx2word, params, is_train=True, lang_model=LM)
            elapsed_time = datetime.datetime.now() - start_time
            print('Epoch %d Done in %s' % (epoch_idx + 1, elapsed_time))
        
        if (epoch_idx + 1) % test_epoch == 0:
            em, f1, loss = run_epoch(model, dev_dataset, 0, idx2word, 
                    params, is_train=False)
            if em < 0.1 and epoch_idx > 5: break
            if max_f1 > f1 - 1e-2 and epoch_idx > 0 and early_stop:
                print('Max em: %.3f, f1: %.3f, epoch: %d' % (max_em, max_f1, max_ep))
                es_cnt += 1
                if epoch_idx > 15:
                    if es_cnt > 3 :
                        print('\nEarly stopping')
                        print('Max f1: %.3f, em: %.3f, epoch: %d' % (max_f1, max_em, max_ep))
                        break
                    else: 
                        # Learning rate decay exponentially
                        print('\nLower learning rate from %f to %f (%d/3)' % (
                            params['learning_rate'], params['learning_rate'] / 2, es_cnt))
                        params['learning_rate'] /= 2
            else:
                es_cnt = 0
                if max_f1 + max_em > f1 + em:
                    max_ep = epoch_idx + 1
                    maxep_em = em
                    maxep_f1 = f1
                    if em > max_em : max_em = em
                    if f1 > max_f1 : max_f1 = f1
                    print('Max f1: %.3f, em: %.3f, epoch: %d' % (max_f1, max_em, max_ep, maxep_em, maxep_f1))
                
                if params['save']:
                    model.save(params['checkpoint_dir'])
    
    model.reset_graph()
    params['learning_rate'] = init_lr
    return max_em, max_f1, max_ep


def sample_parameters(params):
    params['learning_rate'] = float('{0:.5f}'.format(random.randint(1, 1000) * 1e-5))
    # params['dim_rnn_cell'] = random.randint(4, 10) * 10 
    # params['batch_size'] = random.randint(1, 12) * 8
    # params['dim_perspective'] = random.randint(1, 5) * 5
    return params


def write_result(params, em, f1, ep):
    f = open(params['validation_path'], 'a')
    f.write('Model %s\n' % params['model_name'])
    f.write('learning_rate / dim_rnn_cell / batch_size / ' + 
            'dim_perspective / dim_embed_word / context_maxlen\n')
    f.write('[%f / %d / %d / %d / %d / %d]\n' % (params['learning_rate'], 
        params['dim_rnn_cell'], params['batch_size'],
        params['dim_perspective'],
        params['dim_embed_word'], params['context_maxlen']))
    f.write('EM / F1 / EP\n')
    f.write('[%.3f / %.3f / %d]\n\n' % (em, f1, ep))
    f.close()


def main(_):
    # Parse arguments and flags
    expected_version = '1.1'
    saved_params = FLAGS.__flags

    # Make directories
    if not os.path.exists(saved_params['checkpoint_dir']):
        os.makedirs(saved_params['checkpoint_dir'])
    if not os.path.exists(saved_params['summary_dir']):
        os.makedirs(saved_params['summary_dir'])

    # Load dataset once
    train_path = saved_params['train_path']
    dev_path = saved_params['dev_path']
    train_dataset = read_data(train_path, expected_version)
    dev_dataset = read_data(dev_path, expected_version)
    
    """
    Dataset is structured in json format:
        articles (list)
        - paragraphs (list)
            - context
            - qas (list)
                - answers
                - question
                - id 
        - title
    """
    # Preprocess dataset
    whole_dataset = np.append(train_dataset, dev_dataset, axis=0)
    word2idx, idx2word, c_maxlen, q_maxlen, word_maxlen, char2idx, idx2char = \
            build_dict(whole_dataset, saved_params)
    pretrained_glove, word2idx, idx2word = load_glove(word2idx, saved_params)
    if saved_params['context_maxlen'] > 0: 
        c_maxlen = saved_params['context_maxlen']

    train_dataset = preprocess(
            train_dataset, word2idx, c_maxlen, q_maxlen,word_maxlen, char2idx)
    dev_dataset = preprocess(
            dev_dataset, word2idx, c_maxlen, q_maxlen, word_maxlen, char2idx)
    saved_params['context_maxlen'] = c_maxlen
    saved_params['question_maxlen'] = q_maxlen
    saved_params['voca_size'] = len(word2idx)
    saved_params['dim_output'] = c_maxlen
    saved_params['word_maxlen'] = word_maxlen
    saved_params['char_size'] = len(char2idx)

    for model_idx in range(saved_params['validation_cnt']):
        # Copy params, ready for validation
        if saved_params['sample_params']:
            params = sample_parameters(copy.deepcopy(saved_params))
        else:
            params = copy.deepcopy(saved_params)

        # Model name settings
        ymdhms = datetime.datetime.now().strftime('%Y%m%d%H%M%S') 
        params['ymdhms'] = ymdhms
        if params['load']:
            params['model_name'] = params['load_name']
        else:
            params['model_name'] = '%s%d_%s_%d' % (params['mode'],
                    params['context_maxlen'], params['ymdhms'], model_idx)
        
        print('\nModel_%d paramter set' % (model_idx))
        pprint.PrettyPrinter().pprint(params)

        if 'm' == params['mode']:
            my_model = MPCM(params, initializer=[pretrained_glove, word2idx])
        elif 'q' == params['mode']:
            my_model = QL_MPCM(params, initializer=[pretrained_glove, word2idx])
        elif 'b' == params['mode']:
            my_model = Basic(params, initializer=[pretrained_glove, word2idx])
        elif 'bidaf' == params['mode']:
            my_model = BiDAF(params, initializer=[pretrained_glove, word2idx])
        elif 'my_bidaf' == params['mode']:
            my_model = My_BiDAF(params, initializer=[pretrained_glove, word2idx])
        else:
            assert False, "Check your version %s" % params['mode']

        if params['load']:
            my_model.load(params['checkpoint_dir'])
       
        em, f1, max_ep = run(my_model, params, train_dataset, dev_dataset, idx2word)
        write_result(params, em, f1, max_ep)

        if not saved_params['sample_params']:
            break


if __name__ == '__main__':
    tf.app.run()

