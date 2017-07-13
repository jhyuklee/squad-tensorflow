import sys
import tensorflow as tf
import numpy as np
import pprint
import argparse
import datetime
import random
import copy

from model import Basic
from mpcm import MPCM
from ql_mpcm import QL_MPCM
from time import gmtime, strftime
from dataset import read_data, build_dict, load_glove, preprocess
from run import train, test

flags = tf.app.flags
flags.DEFINE_integer('train_epoch', 100, 'Training epoch')
flags.DEFINE_integer('test_epoch', 1, 'Test for every n training epoch')
flags.DEFINE_integer("batch_size", 16, "Size of batch (32)")
flags.DEFINE_integer("dim_perspective", 20, "Maximum number of perspective (20)")
flags.DEFINE_integer("dim_embed_word", 300, "Dimension of word embedding (300)")
flags.DEFINE_integer("dim_rnn_cell", 100, "Dimension of RNN cell (100)")
flags.DEFINE_integer("dim_hidden", 100, "Dimension of hidden layer")
flags.DEFINE_integer("num_paraphrase", 1, "Maximum number of question paraphrasing")
flags.DEFINE_integer("num_action", 4, "Number of action space.")
flags.DEFINE_integer("rnn_layer", 1, "Layer number of RNN ")
flags.DEFINE_integer("context_maxlen", 0, "Predefined context max length")
flags.DEFINE_integer("validation_cnt", 100, "Number of model validation")
flags.DEFINE_float("rnn_dropout", 0.5, "Dropout of RNN cell")
flags.DEFINE_float("hidden_dropout", 0.5, "Dropout rate of hidden layer")
flags.DEFINE_float("embed_dropout", 0.8, "Dropout rate of embedding layer")
flags.DEFINE_float("learning_rate", 0.00162, "Initial learning rate of the optimzier")
flags.DEFINE_float("max_grad_norm", 5.0, "Maximum gradient to clip")
flags.DEFINE_boolean("embed_trainable", False, "True to optimize embedded words")
flags.DEFINE_boolean("test", False, "True to run only iteration 5")
flags.DEFINE_boolean("debug", False, "True to show debug message")
flags.DEFINE_boolean("save", False, "True to save model after testing")
flags.DEFINE_boolean("sample_params", False, "True to sample parameters")
flags.DEFINE_string("model", "m", "b: basic, m: mpcm, q: ql_mpcm")
flags.DEFINE_string('train_path', './data/train-v1.1.json', 'Training dataset path')
flags.DEFINE_string('dev_path', './data/dev-v1.1.json',  'Development dataset path')
flags.DEFINE_string('pred_path', './result/dev-v1.1-pred.json', 'Pred output path')
flags.DEFINE_string('glove_path', \
        ('~/common/glove/glove.6B.'+ str(tf.app.flags.FLAGS.dim_embed_word) +
        'd.txt'), 'embed path')
flags.DEFINE_string('validation_path', './result/validation.txt', 'Validation path')
flags.DEFINE_string('checkpoint_dir', './result/ckpt/', 'Checkpoint directory')
FLAGS = flags.FLAGS


def run(model, params, train_dataset, dev_dataset, idx2word):
    max_em = max_f1 = max_ep = es_cnt = 0
    train_epoch = params['train_epoch']
    test_epoch = params['test_epoch']
    init_lr = params['learning_rate']

    for epoch_idx in range(train_epoch):
        start_time = datetime.datetime.now()
        print("\n[Epoch %d]" % (epoch_idx + 1))
        train(model, train_dataset, epoch_idx + 1, idx2word, params)
        elapsed_time = datetime.datetime.now() - start_time
        print('Epoch %d Done in %s' % (epoch_idx + 1, elapsed_time))
        
        if (epoch_idx + 1) % test_epoch == 0:
            f1, em, loss = test(model, dev_dataset, params)
            
            if max_f1 > f1 - 1e-2 and epoch_idx > 0:
                print('Max f1: %.3f, em: %.3f, epoch: %d' % (max_f1, max_em, max_ep))
                es_cnt += 1
                if es_cnt > 3:
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
                max_ep = max_ep if max_em > em else (epoch_idx + 1)
                max_em = max_em if max_em > em else em 
                max_f1 = max_f1 if max_f1 > f1 else f1
                print('Max f1: %.3f, em: %.3f, epoch: %d' % (max_f1, max_em, max_ep))
                
                if params['save']:
                    model.save(params['checkpoint_dir'])
    
    model.reset_graph()
    params['learning_rate'] = init_lr
    return max_f1, max_em, max_ep


def sample_parameters(params):
    params['learning_rate'] = float('{0:.5f}'.format(random.randint(1, 1000) * 1e-5))
    # params['dim_rnn_cell'] = random.randint(4, 10) * 10 
    # params['batch_size'] = random.randint(1, 12) * 8
    # params['dim_perspective'] = random.randint(1, 5) * 5
    return params


def write_result(params, f1, em, ep):
    f = open(params['validation_path'], 'a')
    f.write('Model %s\n' % params['model'])
    f.write('learning_rate / dim_rnn_cell / batch_size / ' + 
            'dim_perspective / dim_embed_word / context_maxlen\n')
    f.write('[%f / %d / %d / %d / %d / %d]\n' % (params['learning_rate'], 
        params['dim_rnn_cell'], params['batch_size'],
        params['dim_perspective'],
        params['dim_embed_word'], params['context_maxlen']))
    f.write('F1 / EM / EP\n')
    f.write('[%.3f / %.3f / %d]\n\n' % (f1, em, ep))
    f.close()


def main(_):
    # Parse arguments and flags
    expected_version = '1.1'
    saved_params = FLAGS.__flags

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
    word2idx, idx2word, c_maxlen, q_maxlen = build_dict(train_dataset, saved_params)
    pretrained_glove, word2idx, idx2word = load_glove(word2idx, saved_params)
    if saved_params['context_maxlen'] > 0: 
        c_maxlen = saved_params['context_maxlen']

    train_dataset = preprocess(train_dataset, word2idx, c_maxlen, q_maxlen)
    dev_dataset = preprocess(dev_dataset, word2idx, c_maxlen, q_maxlen)
    saved_params['context_maxlen'] = c_maxlen
    saved_params['question_maxlen'] = q_maxlen
    saved_params['voca_size'] = len(word2idx)
    saved_params['dim_output'] = c_maxlen

    for model_idx in range(saved_params['validation_cnt']):
        # Copy params, ready for validation
        if saved_params['sample_params']:
            params = sample_parameters(copy.deepcopy(saved_params))
        else:
            params = copy.deepcopy(saved_params)
        print('\nModel_%d paramter set' % (model_idx))
        pprint.PrettyPrinter().pprint(params)

        # Make model and run experiment
        params['model'] += ('_%d' % model_idx)
        if 'm' in params['model']:
            my_model = MPCM(params, initializer=[pretrained_glove, word2idx])
        elif 'q' in params['model']:
            my_model = QL_MPCM(params, initializer=[pretrained_glove, word2idx])
        elif 'b' in params['model']:
            my_model = Basic(params, initializer=[pretrained_glove, word2idx])
        else:
            assert False, "Check your version %s" % params['model']
       
        f1, em, max_ep = run(my_model, params, train_dataset, dev_dataset, idx2word)
        write_result(params, f1, em, max_ep)

        if not saved_params['sample_params']:
            break


if __name__ == '__main__':
    tf.app.run()

