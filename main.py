import tensorflow as tf
import numpy as np
import pprint
import argparse

from model import RNN
from time import gmtime, strftime
from dataset import read_data, train, test


flags = tf.app.flags
flags.DEFINE_integer('train_epoch', 10, 'Training epoch')
flags.DEFINE_integer("dim_unigram", 82, "Dimension of input, 42 or 82")
flags.DEFINE_integer("dim_output", 127, "Dimension of output, 95 or 127")
flags.DEFINE_integer("max_time_step", 60, "Maximum time step of RNN")
flags.DEFINE_integer("min_grad", -5, "Minimum gradient to clip")
flags.DEFINE_integer("max_grad", 5, "Maximum gradient to clip")
flags.DEFINE_integer("batch_size", 300, "Size of batch")
flags.DEFINE_integer("ngram", 3, "Ngram feature when ensemble = False.")
flags.DEFINE_float("decay_rate", 0.99, "Decay rate of learning rate")
flags.DEFINE_float("decay_step", 100, "Decay step of learning rate")
flags.DEFINE_integer("dim_rnn_cell", 200, "Dimension of RNN cell")
flags.DEFINE_integer("dim_hidden", 200, "Dimension of hidden layer")
flags.DEFINE_integer("dim_embed_unigram", 30, "Dimension of character embedding")
flags.DEFINE_integer("lstm_layer", 1, "Layer number of RNN ")
flags.DEFINE_float("lstm_dropout", 0.5, "Dropout of RNN cell")
flags.DEFINE_float("hidden_dropout", 0.5, "Dropout rate of hidden layer")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate of the optimzier")
flags.DEFINE_boolean("embed", True, "True to embed words")
flags.DEFINE_boolean("embed_trainable", False, "True to optimize embedded words")

flags.DEFINE_string("model_name", "default", "Model name, auto saved as YMDHMS")
flags.DEFINE_string('train_path', './data/train-v1.1.json', 'Training dataset path')
flags.DEFINE_string('dev_path', './data/dev-v1.1.json',  'Development dataset path')
flags.DEFINE_string('pred_path', './result/dev-v1.1-pred.json', 'Prediction output path')
flags.DEFINE_string('checkpoint_dir', './result/ckpt/', 'Checkpoint directory')
FLAGS = flags.FLAGS


def run(model, params, train_dataset, dev_dataset):
    max_em = max_f1 = max_point = 0
    train_epoch = params['train_epoch']

    for epoch_idx in range(train_epoch):
        train(model, params, train_dataset)
    test(model, params, dev_dataset)

    model.reset_graph() 


def main(_):
    expected_version = '1.1'
    parser = argparse.ArgumentParser(
        description='Evaluation for SQuAD ' + expected_version)
    args = parser.parse_args()

    saved_params = FLAGS.__flags
    pprint.PrettyPrinter().pprint(saved_params)
    train_path = saved_params['train_path']
    dev_path = saved_params['dev_path']

    train_dataset = read_data(train_path, expected_version)
    dev_dataset = read_data(dev_path, expected_version)
    params = saved_params.copy()

    """
    Dataset is structured in json format:
        articles (list)
        - paragraphs
            - context
            - qas
                - answers
                - question
                - id 
        - title
    """
    # print(train_dataset[0]['paragraphs'][0])    
    # print(train_dataset[0]['title'])    
    # print(dev_dataset[0]['paragraphs'][0])    
    # print(dev_dataset[0]['title'])

    rnn_model = RNN(params, None)
    run(rnn_model, params, train_dataset, dev_dataset) 


if __name__ == '__main__':
    tf.app.run()

