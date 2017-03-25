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
            cos_d = tf.scan(lambda a, W_k: (W_k * v1)*(W_k * v2), W)
            return tf.reduce_sum(cos_d, axis=1)

        W1 = tf.get_variable('W1', [self.max_perspective, self.dim_hidden],
                initializer=tf.random_normal_initializer())
        W2 = tf.get_variable('W2', [self.max_perspective, self.dim_hidden],
                initializer=tf.random_normal_initializer())
        W3 = tf.get_variable('W3', [self.max_perspective, self.dim_hidden],
                initializer=tf.random_normal_initializer())
        W4 = tf.get_variable('W4', [self.max_perspective, self.dim_hidden],
                initializer=tf.random_normal_initializer())
        W5 = tf.get_variable('W5', [self.max_perspective, self.dim_hidden],
                initializer=tf.random_normal_initializer())
        W6 = tf.get_variable('W6', [self.max_perspective, self.dim_hidden],
                initializer=tf.random_normal_initializer())
       
        # Forward matching
        fw_context_group = tf.split(fw_context, 
                num_or_size_splits=self.context_maxlen, axis=1)
        fw_question_group = tf.split(fw_question, 
                num_or_size_splits=self.question_maxlen, axis=1)

        for context_word in fw_context_group:
            context_word = tf.squeeze(context_word, [1])
            full_list = []
            max_list = []
            mean_list = []
            for question_word in fw_question_group:
                question_word = tf.squeeze(question_word, [1])
                init = tf.zeros([self.max_perspective])
                full_matching = tf.scan(lambda a, w:
                        matching_function(w[0], w[1], W1), (context_word, question_word), init)
                max_matching = tf.scan(lambda a, w:
                        matching_function(w[0], w[1], W3), (context_word, question_word), init)
                mean_matching = tf.scan(lambda a, w:
                        matching_function(w[0], w[1], W5), (context_word, question_word), init)
                full_list.append(full_matching)
                max_list.append(max_matching)
                mean_list.append(mean_matching)

        print('forward matching done')

        # Backward matching
        bw_context_group = tf.split(bw_context, 
                num_or_size_splits=self.context_maxlen, axis=1)
        bw_question_group = tf.split(bw_question, 
                num_or_size_splits=self.question_maxlen, axis=1)

        for context_word in bw_context_group:
            context_word = tf.squeeze(context_word, [1])    
            for question_word in bw_question_group:
                question_word = tf.squeeze(question_word, [1])
                init = tf.zeros([self.max_perspective])
                full_matching = tf.scan(lambda a, w:
                        matching_function(w[0], w[1], W2), (context_word, question_word), init)
                max_matching = tf.scan(lambda a, w:
                        matching_function(w[0], w[1], W4), (context_word, question_word), init)
                mean_matching = tf.scan(lambda a, w:
                        matching_function(w[0], w[1], W6), (context_word, question_word), init)

        print('backword matching done')

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



        
