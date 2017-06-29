import tensorflow as tf
import sys

from mpcm import MPCM
from ops import *


class QL_MPCM(MPCM):
    def __init__(self, params, initializer):
        self.num_paraphrase = params['num_paraphrase']
        super(QL_MPCM, self).__init__(params, initializer)

    def paraphrase_layer(self, question, length, max_length, reuse=None):
        with tf.variable_scope('Paraphrase_Layer', reuse=reuse) as scope:
            cell = lstm_cell(self.dim_rnn_cell, self.rnn_layer, self.rnn_dropout)
            r_inputs = rnn_reshape(question, self.dim_rnn_cell * 2, max_length)
            outputs = rnn_model(r_inputs, length, max_length, cell, self.params)
            # TODO: argmax not differentiable!
            pp_question = tf.argmax(
                    tf.reshape(
                    linear(inputs=outputs,
                        output_dim=self.voca_size,
                        scope='Decode'), [-1, max_length, self.voca_size]), 2)
            
            return pp_question
    
    def optimize_loss(self, start_logits, end_logits):
        batch_size = tf.shape(start_logits)[0]
        start_loss = tf.zeros([batch_size, self.dim_output], tf.float32)
        end_loss = tf.zeros([batch_size, self.dim_output], tf.float32)
        for sl, el in zip(start_logits, end_logits):
            start_loss += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=sl, labels=self.answer_start)) 
            end_loss += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=el, labels=self.answer_end))
            print('loss per question', sl, el)
        self.loss = start_loss + end_loss
        tf.summary.scalar('Loss', self.loss)
        
        print('# Calculating derivatives.. \n')
        self.variables = tf.trainable_variables()
        self.grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, self.variables),
                self.max_grad_norm)
        self.optimize = self.optimizer.apply_gradients(
                zip(self.grads, self.variables), global_step=self.global_step)

    def build_model(self):
        print("Question Learning Model")

        context_embed = dropout(embedding_lookup(
                inputs=self.context,
                voca_size=self.voca_size,
                embedding_dim=self.dim_embed_word, 
                initializer=self.initializer, 
                trainable=self.embed_trainable,
                reuse=True, scope='Word'), self.embed_dropout)

        start_logits = [] 
        end_logits = []
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

            sl, el = self.prediction_layer(aggregates, reuse=(pp_idx>0))
            start_logits.append(sl)
            end_logits.append(el)
            print('# Prediction_layer', sl, el)

        self.optimize_loss(start_logits, end_logits)
        return start_logits, end_logits

