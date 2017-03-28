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
            return outputs
    
    def matching_layer(self, context, question):
        fw_context, bw_context = tf.split(context, num_or_size_splits=2, axis=2)
        fw_question, bw_question = tf.split(question, num_or_size_splits=2, axis=2)
        
        def matching_function(v1, v2, W):
            cos_d = tf.scan(lambda a, W_k: (W_k * v1)*(W_k * v2), W)
            return tf.reduce_sum(cos_d, axis=1)

        W_fw = tf.get_variable('W_fw', [self.max_perspective * 3, self.dim_hidden],
                initializer=tf.random_normal_initializer())
        W_bw = tf.get_variable('W_bw', [self.max_perspective * 3, self.dim_hidden],
                initializer=tf.random_normal_initializer())
       
        fw_context_group = tf.split(fw_context, 
                num_or_size_splits=self.context_maxlen, axis=1)
        bw_context_group = tf.split(bw_context, 
                num_or_size_splits=self.context_maxlen, axis=1)
        fw_question_group = tf.split(fw_question, 
                num_or_size_splits=self.question_maxlen, axis=1)
        bw_question_group = tf.split(bw_question, 
                num_or_size_splits=self.question_maxlen, axis=1)

        for c_idx, (fw_ct, bw_ct) in enumerate(zip(fw_context_group, bw_context_group)):
            fw_ct = tf.squeeze(fw_ct, [1])
            bw_ct = tf.squeeze(bw_ct, [1])
            fw_matching_list = []
            bw_matching_list = []
            for fw_qu, bw_qu in zip(fw_question_group, bw_question_group):
                fw_qu = tf.squeeze(fw_qu, [1])
                bw_qu = tf.squeeze(bw_qu, [1])
                init = tf.zeros([self.max_perspective * 3])
                fw_matching = tf.scan(lambda a, w:
                        matching_function(w[0], w[1], W_fw), (fw_ct, fw_qu), init)
                bw_matching = tf.scan(lambda a, w:
                        matching_function(w[0], w[1], W_bw), (bw_ct, bw_qu), init)
                fw_matching_list.append(fw_matching)
                bw_matching_list.append(bw_matching)

            print('\t', 'fw, bw processing %d/%d' % (c_idx, len(fw_context_group)))
            if c_idx >= 9:
                break

        # print('\t', 'Matching list size:', len(fw_matching_list), len(bw_matching_list))
        # print('\t', 'Matching element size:', fw_matching_list[0], bw_matching_list[0])
       
        def full_matching(sequence, length):
            # TODO: gather 0 or max index of sequence
            """
            sequence: [batch, context_length, question_length, perspective]
            length: [batch]
            """
            """
            indices = tf.concat(axis=1, values=[
                tf.expand_dims(tf.range(0, tf.shape(sequence)[0]), 1),
                tf.expand_dims(tf.range(0, 10), 1),
                tf.expand_dims(length - 1, 1)])
            """
            return sequence

        def max_matching(sequence, length):
            # TODO: gather maximum between 0 ~ max in sequence
            return sequence

        def mean_matching(sequence, length):
            # TODO: gather mean of 0 ~ max in sequence
            return sequence

        fw_matching_total = tf.transpose(tf.stack(fw_matching_list), [1, 0, 2])
        bw_matching_total = tf.transpose(tf.stack(bw_matching_list), [1, 0, 2])
        print('\t', 'fw matching', fw_matching_total)
        print('\t', 'bw matching', bw_matching_total)
        fw_full, fw_max, fw_mean = tf.split(fw_matching_total, num_or_size_splits=3, axis=2)
        bw_full, bw_max, bw_mean = tf.split(bw_matching_total, num_or_size_splits=3, axis=2)
        # print('\t', fw_full, fw_max, fw_mean)
        # print('\t', bw_full, bw_max, bw_mean)
        c_len = 10
        q_len = self.question_maxlen
        fw_full = tf.reshape(fw_full, [-1, c_len, q_len * self.max_perspective])
        '''
        fw_full = tf.reshape(fw_full, [-1, c_len, q_len, self.max_perspective])
        bw_full = tf.reshape(bw_full, [-1, c_len, q_len, self.max_perspective])
        fw_mean = tf.reshape(fw_mean, [-1, c_len, q_len, self.max_perspective])
        bw_mean = tf.reshape(bw_mean, [-1, c_len, q_len, self.max_perspective])
        fw_max = tf.reshape(fw_max, [-1, c_len, q_len, self.max_perspective])
        bw_max = tf.reshape(bw_max, [-1, c_len, q_len, self.max_perspective])
        print(fw_full, fw_max, fw_mean)
        print(bw_full, bw_max, bw_mean)

        total_matching = tf.concat(axis=1, values=[
            full_matching(fw_full, self.question_len),
            full_matching(bw_full, self.question_len),
            max_matching(fw_max, self.question_len),
            max_matching(bw_max, self.question_len),
            mean_matching(fw_mean, self.question_len),
            mean_matching(bw_mean, self.question_len)])
        '''

        return fw_full

    def aggregation_layer(self, inputs, max_length):
        with tf.variable_scope('Aggregation') as scope:
            fw_cell = lstm_cell(self.dim_rnn_cell, self.cell_layer_num, self.lstm_dropout)
            bw_cell = lstm_cell(self.dim_rnn_cell, self.cell_layer_num, self.lstm_dropout)
            r_inputs = rnn_reshape(inputs, self.max_perspective, max_length)
            outputs = bi_rnn_model(r_inputs, None, fw_cell, bw_cell)
            print('\t', 'inputs', inputs)
            print('\t', 'outputs', outputs)
            return tf.reshape(outputs, [-1, 10 * self.dim_rnn_cell * 2])

    def prediction_layer(self, inputs):
        start_logits = linear(inputs=inputs,
            output_dim=self.dim_output, 
            scope='Output_s')

        end_logits = linear(inputs=inputs,
            output_dim=self.dim_output, 
            scope='Output_e')

        return start_logits, end_logits

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
        print('# Filter_layer', context_filtered)

        context_rep = self.representation_layer(context_filtered, self.context_len,
                self.context_maxlen, scope='Context')
        question_rep = self.representation_layer(question_embed, self.question_len,
                self.question_maxlen, scope='Question')
        print('# Representation_layer', context_rep, question_rep)

        matching_vectors = self.matching_layer(context_rep, question_rep)
        print('# Matching_layer', matching_vectors)

        aggregation = self.aggregation_layer(matching_vectors, 10)
        print('# Aggregation_layer', aggregation)
 
        self.start_logits, self.end_logits = self.prediction_layer(aggregation)
        print('# Prediction_layer', self.start_logits, self.end_logits)
 
