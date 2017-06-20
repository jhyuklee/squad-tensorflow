import tensorflow as tf
import sys

from mpcm import MPCM
from ops import *


class QL_MPCM(MPCM):
    def __init__(self, params, initializer):
        self.num_paraphrase = params['num_paraphrase']
        super(QL_MPCM, self).__init__(params, initializer)
    
    def decoder(self, inputs, state, feed_prev=False, reuse=None):
        """
        Args:
            inputs: decoder inputs with size [batch_size, time_steps, input_dim]
            state: hidden state of encoder with size [batch_size, cell_dim]

        Returns:
            decoded: decoded outputs with size [batch_size, time_steps, input_dim]
        """

        with tf.variable_scope('decoder', reuse=reuse):
            # make dummy linear for loop function
            dummy = linear(inputs=tf.constant(1, tf.float32, [100, self.cell_dim]),
                    output_dim=self.input_dim, scope='rnn_decoder/loop_function/Out', reuse=reuse)

            if feed_prev:
                def loop_function(prev, i):
                    next = tf.argmax(linear(inputs=prev,
                        output_dim=self.input_dim,
                        scope='Out', reuse=True), 1)
                    return tf.one_hot(next, self.input_dim)
            else:
                loop_function = None

            cell = lstm_cell(self.cell_dim, self.cell_layer_num, self.cell_keep_prob)
            inputs = tf.one_hot(inputs, self.input_dim)
            inputs_t = tf.unstack(tf.transpose(inputs, [1, 0, 2]), self.max_time_step)
            outputs, states = tf.contrib.legacy_seq2seq.rnn_decoder(inputs_t, state, cell, loop_function)
            outputs_t = tf.transpose(tf.stack(outputs), [1, 0, 2])
            outputs_tr = tf.reshape(outputs_t, [-1, self.cell_dim])
            decoded = linear(inputs=outputs_tr,
                    output_dim=self.input_dim, scope='rnn_decoder/loop_function/Out', reuse=True)
            return decoded

    def paraphrase_layer(self, question, length, max_length, reuse=None):
        with tf.variable_scope('Paraphrase_Layer', reuse=reuse) as scope:
            cell = lstm_cell(self.dim_rnn_cell, self.rnn_layer, self.rnn_dropout)
            r_inputs = rnn_reshape(question, self.dim_rnn_cell, max_length)
            outputs = rnn_model(r_inputs, length, max_length, cell, self.params)
            pp_question = tf.argmax(
                    tf.reshape(
                    linear(inputs=outputs,
                        output_dim=self.voca_size,
                        scope='Decode'), [-1, max_length, self.voca_size]), 2)
            
            return pp_question

    def build_model(self):
        print("Question Learning Model")

        context_embed = dropout(embedding_lookup(
                inputs=self.context,
                voca_size=self.voca_size,
                embedding_dim=self.dim_embed_word, 
                initializer=self.initializer, 
                trainable=self.embed_trainable,
                reuse=True, scope='Word'), self.embed_dropout)

        for pp_idx in range(self.num_paraphrase + 1):
            if pp_idx > 0:
                paraphrased = self.paraphrase_layer(question_rep, 
                        self.question_len, self.question_maxlen, reuse=(pp_idx>1))
                print('# Paraphrase_layer %d' % (pp_idx), paraphrased)
                # TODO: Add paraphrase loss here
            else:
                paraphrased = self.question
            
            question_embed = dropout(embedding_lookup(
                    inputs=paraphrased,
                    voca_size=self.voca_size,
                    embedding_dim=self.dim_embed_word,
                    initializer=self.initializer,
                    trainable=self.embed_trainable,
                    reuse=True, scope='Word'), self.embed_dropout)

            question_rep = self.representation_layer(question_embed, self.question_len,
                    self.question_maxlen, scope='Question', reuse=(pp_idx>0))
            
            context_filtered = self.filter_layer(context_embed, question_embed, reuse=(pp_idx>0))
            print('# Filter_layer', context_filtered)
          
            context_rep = self.representation_layer(context_filtered, self.context_len,
                    self.context_maxlen, scope='Context', reuse=(pp_idx>0))
            print('# Representation_layer', context_rep, question_rep)

            matchings = self.matching_layer(context_rep, question_rep, reuse=(pp_idx>0))
            print('# Matching_layer', matchings)

            aggregates = self.aggregation_layer(matchings, self.context_maxlen,
                    self.context_len, reuse=(pp_idx>0))
            print('# Aggregation_layer', aggregates)        

            start_logits, end_logits = self.prediction_layer(aggregates, reuse=(pp_idx>0))
            print('# Prediction_layer', start_logits, end_logits)
            # TODO: Add answer loss here

        return start_logits, end_logits


