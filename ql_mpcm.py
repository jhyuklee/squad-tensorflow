import tensorflow as tf
import sys

from mpcm import MPCM
from ops import *


class QL_MPCM(MPCM):
    def __init__(self, params, initializer):
        self.num_paraphrase = params['num_paraphrase']
        self.rewards = []
        self.paraphrases = []
        self.paraphrase_logits = []
        self.paraphrase_optimize = []
        for _ in range(self.num_paraphrase):
            self.rewards.append(tf.placeholder(tf.float32, [None]))
        super(QL_MPCM, self).__init__(params, initializer)

    def paraphrase_layer(self, question, length, max_length, reuse=None):
        with tf.variable_scope('Paraphrase_Layer', reuse=reuse) as scope:
            cell = lstm_cell(self.dim_rnn_cell, self.rnn_layer, self.rnn_dropout)
            r_inputs = rnn_reshape(question, self.dim_rnn_cell * 2, max_length)
            outputs = rnn_model(r_inputs, length, max_length, cell, self.params)
           
            pp_logits = tf.reshape(linear(inputs=outputs,
                output_dim=self.voca_size,
                scope='Decode'), [-1, max_length, self.voca_size])

            # Not argamx but sample?
            pp_question = tf.argmax(pp_logits, 2)
            return pp_question, pp_logits

    def optimize_paraphrased(self, pp_sample, pp_logits, paraphrase_cnt):
        print("# Calculating Paraphrased Loss")
        reward = self.rewards[paraphrase_cnt - 1]
        log_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=pp_logits, labels=pp_sample)
        policy_grads = tf.expand_dims(reward, -1) * log_loss
        policy_grads = tf.reduce_mean(tf.reduce_sum(policy_grads))
        optimize = self.optimizer.minimize(policy_grads)
        self.paraphrase_optimize.append(optimize)

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
                paraphrased, pp_logits = self.paraphrase_layer(question_rep, 
                        self.question_len, self.question_maxlen, reuse=(pp_idx>1))
                self.paraphrases.append(paraphrased)
                print('# Paraphrase_layer %d' % (pp_idx), paraphrased)
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

            sl, el= self.prediction_layer(aggregates, reuse=(pp_idx>0))
            print('# Prediction_layer', sl, el)

            if pp_idx > 0:
                self.paraphrase_logits.append([sl, el])
                self.optimize_paraphrased(paraphrased, pp_logits, pp_idx)
            else:
             self.start_logits, self.end_logits = [sl, el]
             self.optimize_loss(self.start_logits, self.end_logits)

