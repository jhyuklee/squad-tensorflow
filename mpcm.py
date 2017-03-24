import tensorflow as tf

from model import Basic
from ops import *


class MPCM(Basic):
    def __init__(self, params, initializer):
        self.max_perspective = params['max_perspective']
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

    def representation_layer(self, inputs, length, max_length, scope=None):
        with tf.variable_scope('Representation/' + scope) as scope:
            fw_cell = lstm_cell(self.dim_rnn_cell, self.cell_layer_num, self.lstm_dropout)
            bw_cell = lstm_cell(self.dim_rnn_cell, self.cell_layer_num, self.lstm_dropout)
            r_inputs = rnn_reshape(inputs, self.dim_embed_word, max_length)
            outputs = bi_rnn_model(r_inputs, length, fw_cell, bw_cell)
            # TODO: gather until actual length
            return outputs
    
    def matching_layer(self, context, question):
        fw_context, bw_context = tf.split(context, num_or_size_splits=2, axis=2)
        fw_question, bw_question = tf.split(question, num_or_size_splits=2, axis=2)
        
        def matching_function(v1, v2, W):
            print('matching function')
            print(v1, v2, W)
            print('matching function')
            print(tf.scan(lambda a, W_k: W_k * v1, W))
            print('scan WK')
            return tf.scan(lambda a, W_k: tf.reduce_sum(tf.multiply(W_k * v1, W_k * v2)), W)

        # Full-matching
        W1 = tf.get_variable('W1', [self.max_perspective, self.dim_hidden],
                initializer=tf.random_normal_initializer())
        W2 = tf.get_variable('W2', [self.max_perspective, self.dim_hidden],
                initializer=tf.random_normal_initializer())
        
        fw_context_group = tf.split(fw_context, num_or_size_splits=self.context_maxlen, axis=1)
        print('split context', len(fw_context_group), tf.squeeze(fw_context_group[0], [1]))

        full_matching = tf.scan(lambda a, h: matching_function(tf.squeeze(h[0], [0]),
                tf.squeeze(h[1], [0]), W1), fw_context_group)
        print('perspective', full_matching)

        # TODO: Maxpooling-matching
        # TODO: Meanpooling-matching
        
        return 'matching_layer'

    def aggregation_layer(self):

        return 'aggregation_layer'

    def prediction_layer(self):

        return 'prediction'

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
                self.context_maxlen, scope='Context')
        question_rep = self.representation_layer(question_embed, self.question_len,
                self.question_maxlen, scope='Question')
        print('representation_layer', context_rep, question_rep)

        matching_vectors = self.matching_layer(context_rep, question_rep)



        
