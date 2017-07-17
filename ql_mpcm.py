import tensorflow as tf
import sys

from mpcm import MPCM
from ops import *


class QL_MPCM(MPCM):
    def __init__(self, params, initializer):
        self.num_paraphrase = params['num_paraphrase']
        self.num_action = params['num_action']
        self.rewards = []
        self.baselines = []
        self.paraphrases = []
        self.pp_logits = []
        self.pp_loss = []
        self.pp_optimize = []
        self.action_samples = []
        for _ in range(self.num_paraphrase):
            self.rewards.append(tf.placeholder(tf.float32, [None]))
            self.baselines.append(tf.placeholder(tf.float32, [None]))
            self.paraphrases.append(tf.placeholder(tf.int32, 
                [None, params['question_maxlen']]))
        super(QL_MPCM, self).__init__(params, initializer)
    
    def paraphrase_layer(self, question, c_state, length, max_length, reuse=None):
        with tf.variable_scope('Paraphrase_Layer', reuse=reuse) as scope:
            batch_size = tf.shape(length)[0]
            weights = tf.get_variable('out_w', [self.dim_rnn_cell * 2, self.num_action],
                                      initializer=tf.random_normal_initializer())
            biases = tf.get_variable('out_b', [self.num_action],
                                     initializer=tf.constant_initializer(0.0))
           
            # Bidirectional
            fw_cell = lstm_cell(self.dim_rnn_cell, self.rnn_layer, self.rnn_dropout)
            bw_cell = lstm_cell(self.dim_rnn_cell, self.rnn_layer, self.rnn_dropout)
            outputs, state = bi_rnn_model(question, length, fw_cell, bw_cell, 
                    c_state[0], c_state[1])
            outputs_t = tf.reshape(outputs, [-1, self.dim_rnn_cell * 2])
    
            # Unidirectional
            """
            zero_state = cell.zero_state(batch_size, tf.float32)
            r_inputs = rnn_reshape(question, self.dim_rnn_cell * 2, max_length)
            cell = lstm_cell(self.dim_rnn_cell, self.rnn_layer, self.rnn_dropout)
            outputs, state = tf.contrib.legacy_seq2seq.rnn_decoder(
                    r_inputs, c_state[0], cell)
            outputs_t = tf.reshape(tf.stack(outputs), [-1, self.dim_rnn_cell])
            """

            action_logits = tf.reshape(tf.matmul(outputs_t, weights) + biases,
                    [-1, max_length, self.num_action])

            action_sample = tf.argmax(action_logits, 2)
            return action_sample, action_logits

    def optimize_pp(self, action_sample, action_logits, paraphrase_cnt):
        print("# Calculating Paraphrased Loss\n")
        reward = self.rewards[paraphrase_cnt]
        baseline = self.baselines[paraphrase_cnt]

        score = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=action_logits, labels=action_sample)
        advantage = reward - baseline
        policy_loss = tf.reduce_mean(
                tf.reduce_sum(tf.expand_dims(advantage, -1) * score))
        
        self.policy_params = [p for p in tf.trainable_variables()
                if (('Paraphrase_Layer' in p.name))]
                # or ('Representation_Layer' in p.name))]
        # print([p.name for p in self.policy_params])
        policy_grads, _ = tf.clip_by_global_norm(tf.gradients(
            policy_loss, self.policy_params), self.max_grad_norm)
        optimize = self.optimizer.apply_gradients(zip(policy_grads, self.policy_params),
                global_step=self.global_step)
        self.pp_optimize.append(optimize)
        self.pp_loss.append(policy_loss)

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
                action_sample, action_logits = self.paraphrase_layer(
                        question_rep, c_state,
                        self.question_len, self.question_maxlen, reuse=(pp_idx>1))
                print('# Paraphrase_layer %d' % (pp_idx), action_sample)
                paraphrased = self.paraphrases[pp_idx-1]
                self.action_samples.append(action_sample)
            else:
                paraphrased = self.question
            
            question_embed = dropout(embedding_lookup(
                    inputs=paraphrased,
                    voca_size=self.voca_size,
                    embedding_dim=self.dim_embed_word,
                    initializer=self.initializer,
                    trainable=self.embed_trainable,
                    reuse=True, scope='Word'), self.embed_dropout)

            question_rep, _ = self.representation_layer(question_embed, 
                    self.question_len, self.question_maxlen,
                    scope='Question', reuse=(pp_idx>0))
            
            context_filtered = self.filter_layer(context_embed, question_embed, 
                    reuse=(pp_idx>0))
            print('# Filter_layer', context_filtered)
          
            context_rep, c_state = self.representation_layer(context_filtered, 
                    self.context_len,
                    self.context_maxlen, scope='Context', reuse=(pp_idx>0))
            print('# Representation_layer', context_rep, question_rep)

            matchings = self.matching_layer(context_rep, 
                    question_rep, reuse=(pp_idx>0))
            print('# Matching_layer', matchings)

            aggregates = self.aggregation_layer(matchings, self.context_maxlen,
                    self.context_len, reuse=(pp_idx>0))
            print('# Aggregation_layer', aggregates) 

            sl, el= self.prediction_layer(aggregates, reuse=(pp_idx>0))
            print('# Prediction_layer', sl, el)

            if pp_idx > 0:
                self.pp_logits.append([sl, el])
                self.optimize_pp(action_sample, action_logits, pp_idx-1)
            else:
                self.start_logits, self.end_logits = [sl, el]
                general_params = [p for p in tf.trainable_variables() 
                        if 'Paraphrase_Layer' not in p.name]
                self.optimize_loss(self.start_logits, self.end_logits, general_params)

