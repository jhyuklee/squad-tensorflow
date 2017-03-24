import tensorflow as tf

from model import Basic
from ops import *


class MPCM(Basic):
    def __init__(self, params, initializer):
        super(MPCM, self).__init__(params, initializer)

    def filter_layer(self, context, question):
        c_norm = tf.norm(context, axis=2, keep_dims=True)
        q_norm = tf.norm(question, axis=2, keep_dims=True)
        n_context = context / c_norm
        n_question = question / q_norm
        tr_question = tf.transpose(n_question, [0, 2, 1])
        similarity = tf.matmul(n_context, tr_question)
        max_similarity = tf.reduce_max(similarity, 2, keep_dims=True)
        return tf.multiply(context, max_similarity)

    def representation_layer(self, inputs, length, max_length):
        fw_cell = lstm_cell(self.dim_rnn_cell, self.cell_layer_num, self.lstm_dropout)
        bw_cell = lstm_cell(self.dim_rnn_cell, self.cell_layer_num, self.lstm_dropout)
        r_inputs = rnn_reshape(inputs, self.dim_embed_word, max_length)
        outputs = bi_rnn_model(inputs, length, fw_cell, bw_cell)
        print('rep input', inputs)
        print('rep output', outputs)
        return 'representation_layer'


    def build_model(self):
        print("## Building MPCM model ###")
         
        context_embed = embedding_lookup(
                inputs=self.context,
                voca_size=self.dim_word,
                embedding_dim=self.dim_embed_word, 
                initializer=self.initializer, 
                trainable=self.embed_trainable, scope='Word')

        question_embed = embedding_lookup(
                inputs=self.question,
                voca_size=self.dim_word,
                embedding_dim=self.dim_embed_word,
                initializer=self.initializer,
                trainable=self.embed_trainable,
                reuse=True, scope='Word')

        context_filtered = self.filter_layer(context_embed, question_embed)
        print('filter_layer', context_filtered)

        context_rep = self.representation_layer(context_filtered, self.context_len,
                self.context_maxlen)
        question_rep = self.representation_layer(question_embed, self.question_len,
                self.question_maxlen)

        

        
        




