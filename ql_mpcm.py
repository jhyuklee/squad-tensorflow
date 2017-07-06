import tensorflow as tf
import sys

from mpcm import MPCM
from ops import *


class QL_MPCM(MPCM):
    def __init__(self, params, initializer):
        self.num_paraphrase = params['num_paraphrase']
        self.rewards = []
        self.baselines = []
        self.paraphrases = []
        self.paraphrase_logits = []
        self.paraphrase_optimize = []
        for _ in range(self.num_paraphrase):
            self.rewards.append(tf.placeholder(tf.float32, [None]))
            self.baselines.append(tf.placeholder(tf.float32, [None]))
        super(QL_MPCM, self).__init__(params, initializer)

    def paraphrase_layer(self, question_state, length, max_length, reuse=None):
        with tf.variable_scope('Word', reuse=True):
            embedding_table = tf.get_variable("embed", dtype=tf.float32)
        
        with tf.variable_scope('Paraphrase_Layer', reuse=reuse) as scope:
            batch_size = tf.shape(length)[0]
            cell = lstm_cell(self.dim_rnn_cell, self.rnn_layer, self.rnn_dropout)
            dummy_input = tf.zeros([batch_size, max_length, self.dim_embed_word])
            r_inputs = rnn_reshape(dummy_input, self.dim_embed_word, max_length)
            
            weights = tf.get_variable('out_w', [self.dim_rnn_cell, self.voca_size],
                                      initializer=tf.random_normal_initializer())
            biases = tf.get_variable('out_b', [self.voca_size],
                                     initializer=tf.constant_initializer(0.0))

            def loop_function(prev, i):
                word_idx = tf.argmax(tf.matmul(prev, weights) + biases, 1)
                embed_word = tf.nn.embedding_lookup(embedding_table, word_idx)
                return embed_word
           
            # print('state passed', question_state[0])
            # print('zero state', cell.zero_state(batch_size, tf.float32))
            zero_state = cell.zero_state(batch_size, tf.float32)
            outputs, state = tf.contrib.legacy_seq2seq.rnn_decoder(r_inputs, 
                    question_state[0], cell, loop_function)
            outputs_t = tf.reshape(tf.stack(outputs), [-1, self.dim_rnn_cell])
            pp_logits = tf.reshape(tf.matmul(outputs_t, weights) + biases,
                    [-1, max_length, self.voca_size])

            # Not argamx but softmax approximation?
            pp_sample = tf.argmax(pp_logits, 2)
            return pp_sample, pp_logits

    def optimize_paraphrased(self, pp_sample, pp_logits, paraphrase_cnt):
        print("# Calculating Paraphrased Loss\n")
        reward = self.rewards[paraphrase_cnt - 1]
        baseline = self.baselines[paraphrase_cnt - 1]

        score = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=pp_logits, labels=pp_sample)
        advantage = reward - baseline
        self.policy_loss = tf.reduce_mean(tf.reduce_sum(tf.expand_dims(advantage, -1) * score))

        # TODO: Bag Of Words Loss
        question_bow = tf.reduce_sum(tf.one_hot(self.question, self.voca_size), axis=1)

        bow_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=pp_logits, labels=self.question)
        # self.policy_loss += tf.reduce_mean(tf.reduce_sum(bow_loss))
        
        self.policy_params = [p for p in tf.trainable_variables()
                if 'Paraphrase_Layer' in p.name]
        # print([p.name for p in self.policy_params])
        policy_grads, _ = tf.clip_by_global_norm(tf.gradients(
            self.policy_loss, self.policy_params), self.max_grad_norm)
        optimize = self.optimizer.apply_gradients(zip(policy_grads, self.policy_params),
                global_step=self.global_step)
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
                paraphrased, pp_logits = self.paraphrase_layer(q_state, 
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

            question_rep, q_state = self.representation_layer(question_embed, 
                    self.question_len, self.question_maxlen,
                    scope='Question', reuse=(pp_idx>0))
            
            context_filtered = self.filter_layer(context_embed, question_embed, reuse=(pp_idx>0))
            print('# Filter_layer', context_filtered)
          
            context_rep, _ = self.representation_layer(context_filtered, self.context_len,
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
                general_params = [p for p in tf.trainable_variables() 
                        if 'Paraphrase_Layer' not in p.name]
                self.optimize_loss(self.start_logits, self.end_logits, general_params)

