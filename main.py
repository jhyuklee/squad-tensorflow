import sys
import tensorflow as tf
import numpy as np
import pprint
import argparse

from model import Basic
from mpcm import MPCM
from time import gmtime, strftime
from dataset import read_data, build_dictionary, load_glove, preprocess
from run import train, test

flags = tf.app.flags
flags.DEFINE_integer('train_epoch', 100, 'Training epoch')
flags.DEFINE_integer('test_epoch', 3, 'Test for every n training epoch')
flags.DEFINE_integer("min_voca", 3, "Minimum frequency of word")
flags.DEFINE_integer("min_grad", -5, "Minimum gradient to clip")
flags.DEFINE_integer("max_grad", 5, "Maximum gradient to clip")
flags.DEFINE_integer("batch_size", 2, "Size of batch")
flags.DEFINE_integer("dim_perspective", 10, "Maximum number of perspective")
flags.DEFINE_integer("dim_embed_word", 50, "Dimension of word embedding")
flags.DEFINE_integer("dim_rnn_cell", 40, "Dimension of RNN cell")
flags.DEFINE_integer("dim_hidden", 200, "Dimension of hidden layer")
flags.DEFINE_integer("rnn_layer", 1, "Layer number of RNN ")
flags.DEFINE_float("rnn_dropout", 0.5, "Dropout of RNN cell")
flags.DEFINE_float("hidden_dropout", 0.5, "Dropout rate of hidden layer")
flags.DEFINE_float("embed_dropout", 0.8, "Dropout rate of embedding layer")
flags.DEFINE_float("learning_rate", 1e-3, "Learning rate of the optimzier")
flags.DEFINE_float("decay_rate", 0.99, "Decay rate of learning rate")
flags.DEFINE_float("decay_step", 100, "Decay step of learning rate")
flags.DEFINE_boolean("embed", True, "True to embed words")
flags.DEFINE_boolean("embed_pretrained", True, "True to use pretrained embed words")
flags.DEFINE_boolean("embed_trainable", False, "True to optimize embedded words")
flags.DEFINE_boolean("test", False, "True to max iteration 5")
flags.DEFINE_boolean("debug", False, "True to show debug message")

flags.DEFINE_string("model", "m", "b: basic, m: mpcm")
flags.DEFINE_string('train_path', './data/train-v1.1.json', 'Training dataset path')
flags.DEFINE_string('dev_path', './data/dev-v1.1.json',  'Development dataset path')
flags.DEFINE_string('pred_path', './result/dev-v1.1-pred.json', 'Prediction output path')
flags.DEFINE_string('glove_path', \
        '~/embed_data/glove.6B.'+ str(tf.app.flags.FLAGS.dim_embed_word) +'d.txt', 'embed path')
flags.DEFINE_string('checkpoint_dir', './result/ckpt/', 'Checkpoint directory')
FLAGS = flags.FLAGS


def run(model, params, train_dataset, dev_dataset):
    max_em = max_f1 = max_point = 0
    train_epoch = params['train_epoch']
    test_epoch = params['test_epoch']

    for epoch_idx in range(train_epoch):
        print("\nEpoch %d" % (epoch_idx + 1))
        train(model, train_dataset, params)
        if epoch_idx % test_epoch == 0:
            test(model, dev_dataset, params)
    
    test(model, dev_dataset, params)
    model.reset_graph() 


def main(_):
    # Parse arguments and flags
    expected_version = '1.1'
    saved_params = FLAGS.__flags
    pprint.PrettyPrinter().pprint(saved_params)

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
    dictionary, rev_dict, c_maxlen, q_maxlen = build_dictionary(train_dataset, saved_params)

    train_dataset = preprocess(train_dataset, dictionary, c_maxlen, q_maxlen)
    dev_dataset = preprocess(dev_dataset, dictionary, c_maxlen, q_maxlen)
    saved_params['context_maxlen'] = c_maxlen
    saved_params['question_maxlen'] = q_maxlen
    saved_params['voca_size'] = len(dictionary)
    saved_params['dim_output'] = c_maxlen
    
    if saved_params['embed_pretrained']:
        pretrained_glove = load_glove(dictionary, saved_params)
    else:
        pretrained_glove = None

    # Copy params, ready for validation
    # TODO: Validation parameters
    params = saved_params.copy()

    # Make model and run experiment
    if saved_params['model'] == 'm':
        my_model = MPCM(params, initializer=pretrained_glove)
    elif saved_params['model'] == 'b':
        my_model = Basic(params, initializer=pretrained_glove)
    else:
        assert False, "Check your version %s" % saved_params['model']
    run(my_model, params, train_dataset, dev_dataset) 


if __name__ == '__main__':
    tf.app.run()

