import tensorflow as tf
import sys

from mpcm import MPCM
from ops import *


class QL_MPCM(MPCM):
    def __init__(self, params, initializer):
        self.num_paraphrase = params['num_paraphrase']
        self.num_action = params['num_action']
        self.init_exp = params['init_exp']
        self.final_exp = params['final_exp']
        self.exploration = self.init_exp
        self.rewards = []
        self.baselines = []
        self.paraphrases = []
        self.pp_logits = []
        self.pp_loss = []
        self.pp_optimize = []
        self.taken_actions = []
        self.action_probs = []
        for _ in range(self.num_paraphrase):
            self.rewards.append(tf.placeholder(tf.float32, [None]))
            self.baselines.append(tf.placeholder(tf.float32, [None]))
            self.taken_actions.append(tf.placeholder(tf.int32, 
                [None, params['question_maxlen']]))
            self.paraphrases.append(tf.placeholder(tf.int32, 
                [None, params['question_maxlen']]))
        super(QL_MPCM, self).__init__(params, initializer)
    
    def paraphrase_layer(self, question, c_state, length, max_length, reuse=None):
        with tf.variable_scope('Paraphrase_Layer', reuse=reuse) as scope:
            weights = tf.get_variable('out_w', 
                    [self.dim_rnn_cell * 2, self.num_action],
                    initializer=tf.random_normal_initializer())
            biases = tf.get_variable('out_b', 
                    [self.num_action],                 
                    initializer=tf.constant_initializer(0.0))
           
            # Bidirectional
            fw_cell = lstm_cell(self.dim_rnn_cell, self.rnn_layer, self.rnn_dropout)
            bw_cell = lstm_cell(self.dim_rnn_cell, self.rnn_layer, self.rnn_dropout)
            outputs, state = bi_rnn_model(question, length, fw_cell, bw_cell, 
                    c_state[0], c_state[1])
            outputs_t = tf.reshape(outputs, [-1, self.dim_rnn_cell * 2])
            action_logit = tf.reshape(tf.matmul(outputs_t, weights) + biases,
                    [-1, max_length, self.num_action])

            # TODO: Not argmax but multinomial
            action_sample = tf.argmax(action_logit, 2)
            return action_sample, action_logit

    def optimize_pp(self, action_logit, paraphrase_cnt):
        print("# Calculating Paraphrased Loss\n")
        reward = self.rewards[paraphrase_cnt]
        baseline = self.baselines[paraphrase_cnt]
        taken_action = self.taken_actions[paraphrase_cnt]
        # Mask actions with action length

        # Add regularizer? (reg_loss)
        q_mask = tf.cast(tf.sequence_mask(self.question_len, self.question_maxlen),
                tf.float32)
        pg_loss = tf.contrib.seq2seq.sequence_loss(
                logits=action_logit,
                targets=taken_action,
                weights=q_mask,
                average_across_batch=False)

        # Per batch advantage is straight forward..
        advantage = tf.reduce_sum(reward - baseline) 
       
        # Optimize only paraphrase module
        self.policy_params = [p for p in tf.trainable_variables()
                if (('Paraphrase_Layer' in p.name))]
                # or ('Representation_Layer' in p.name))]
        # print([p.name for p in self.policy_params])

        self.policy_gradients = self.optimizer.compute_gradients(pg_loss, 
                var_list=self.policy_params)

        for i, (grad, var) in enumerate(self.policy_gradients):
            if grad is not None:
                # grad = tf.clip_by_global_norm(grad, self.max_grad_norm)
                self.policy_gradients[i] = (grad * advantage, var)

        optimize = self.optimizer.apply_gradients(self.policy_gradients,
                global_step=self.global_step)

        self.pp_optimize.append(optimize)
        self.pp_loss.append(pg_loss)

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
                _, action_logit = self.paraphrase_layer(
                        question_rep, c_state,
                        self.question_len, self.question_maxlen, reuse=(pp_idx>1))
                print('# Paraphrase_layer %d' % (pp_idx), action_logit)
                paraphrased = self.paraphrases[pp_idx-1]
                self.action_probs.append(tf.nn.softmax(action_logit))
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
                self.optimize_pp(action_logit, pp_idx-1)
            else:
                self.start_logits, self.end_logits = [sl, el]
                general_params = [p for p in tf.trainable_variables() 
                        if 'Paraphrase_Layer' not in p.name]
                self.optimize_loss(self.start_logits, self.end_logits, general_params)

