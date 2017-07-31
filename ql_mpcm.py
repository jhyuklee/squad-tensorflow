import tensorflow as tf
import sys

from mpcm import MPCM
from ops import *


class QL_MPCM(MPCM):
    def __init__(self, params, initializer):
        self.num_paraphrase = params['num_paraphrase']
        self.dim_action = params['dim_action']
        self.max_action = params['max_action']
        self.init_exp = params['init_exp']
        self.final_exp = params['final_exp']
        self.pp_dim_rnn_cell = params['pp_dim_rnn_cell']
        self.pp_rnn_layer = params['pp_rnn_layer']
        self.policy_q = params['policy_q']
        self.policy_c = params['policy_c']
        self.similarity_q = params['similarity_q']
        self.similarity_c = params['similarity_c']
        self.exploration = self.init_exp
        self.advantages = []
        self.paraphrases = [] # paraphrased question after applying action rules
        self.pp_logits = [] # logits after paraphrasing
        self.pp_loss = []
        self.pp_optimize = []
        self.taken_actions = []
        self.action_probs = []
        for _ in range(self.num_paraphrase):
            self.advantages.append(tf.placeholder(tf.float32, [None]))
            self.taken_actions.append(tf.placeholder(tf.int32, 
                [None, params['question_maxlen']]))
            self.paraphrases.append(tf.placeholder(tf.int32, 
                [None, params['question_maxlen']]))
        super(QL_MPCM, self).__init__(params, initializer)

    def similarity_layer(self, context, question, context_rep, reuse=None):
        with tf.variable_scope('Similarity_Layer', reuse=reuse) as scope:
            c_norm = tf.sqrt(
                    tf.maximum(tf.reduce_sum(tf.square(context), axis=-1), 1e-6))
            q_norm = tf.sqrt(
                    tf.maximum(tf.reduce_sum(tf.square(question), axis=-1), 1e-6))
            n_context = context / tf.expand_dims(c_norm, -1)
            n_question = question / tf.expand_dims(q_norm, -1)
            tr_question = tf.transpose(n_question, [0, 2, 1])
            sim_mat_dim = (self.dim_embed_word if self.similarity_q == 'e'
                    else self.dim_rnn_cell * 2)
            sim_mat = tf.get_variable('sim_mat',
                    initializer=tf.eye(sim_mat_dim), dtype=tf.float32)
            # sim_mat = tf.get_variable('sim_mat',
            #         initializer=tf.random_uniform(
            #              [sim_mat_dim, sim_mat_dim], -1, 1), dtype=tf.float32)
            b_sim_mat = tf.scan(lambda a, x: tf.identity(sim_mat), 
                    context, tf.zeros([sim_mat_dim, sim_mat_dim], 
                    dtype=tf.float32)) 
            tmp_cont_sim = tf.matmul(n_context, b_sim_mat)
            self.similarity = tf.matmul(tmp_cont_sim, tr_question)
            self.c_sim = tf.argmax(tf.transpose(self.similarity, [0, 2, 1]), axis=2)
        
        if self.policy_c == 'e':
            selected_context = tf.scan(lambda a, x: tf.gather(x[0], x[1]),
                    (self.context, self.c_sim), 
                    tf.zeros([self.question_maxlen], dtype=tf.int32))
            candidate = dropout(embedding_lookup(
                    inputs=selected_context,
                    voca_size=self.voca_size,
                    embedding_dim=self.dim_embed_word, 
                    initializer=self.initializer, 
                    trainable=self.embed_trainable,
                    reuse=True, scope='Word'), self.embed_dropout)
        else:
            candidate = tf.scan(lambda a, x: tf.gather(x[0], x[1]),
                    (context_rep, self.c_sim), 
                    tf.zeros([self.question_maxlen, self.dim_rnn_cell * 2], 
                        dtype=tf.float32))

        return candidate
    
    def paraphrase_layer(self, question, c_state, length, max_length, 
            candidate=None, reuse=None):
        with tf.variable_scope('Paraphrase_Layer', reuse=reuse) as scope:
            weights = tf.get_variable('out_w', 
                    [self.pp_dim_rnn_cell * 2, self.dim_action],
                    initializer=tf.random_normal_initializer())
            biases = tf.get_variable('out_b', 
                    [self.dim_action],                 
                    initializer=tf.constant_initializer(0.0))
            
            # Concat question and context [q, c_sim, c_fb]
            if candidate is not None:
                c_fb = tf.concat(axis=1, values=[c_state[0][0][1], c_state[1][0][1]])
                c_fb = tf.tile(tf.expand_dims(c_fb, axis=1),
                        [1, self.question_maxlen, 1])
                # question = tf.concat(axis=2, values=[question, candidate, c_fb])
                question = tf.concat(axis=2, values=[question, candidate])
           
            # Bidirectional
            fw_cell = lstm_cell(
                    self.pp_dim_rnn_cell, self.pp_rnn_layer, self.rnn_dropout)
            bw_cell = lstm_cell(
                    self.pp_dim_rnn_cell, self.pp_rnn_layer, self.rnn_dropout)
            outputs, state = bi_rnn_model(question, length, fw_cell, bw_cell)
            #        c_state[0], c_state[1])
            outputs_t = tf.reshape(outputs, [-1, self.pp_dim_rnn_cell * 2])
            action_logit = tf.reshape(tf.matmul(outputs_t, weights) + biases,
                    [-1, max_length, self.dim_action])

            action_sample = tf.multinomial(
                    tf.reshape(action_logit, [-1, self.dim_action]), 1)
            action_sample = tf.reshape(action_sample, 
                    [-1, self.question_maxlen, self.dim_action])
            return action_sample, action_logit

    def optimize_pp(self, action_logit, paraphrase_cnt):
        print("# Calculating Paraphrased Loss\n")
        advantage = self.advantages[paraphrase_cnt]
        taken_action = self.taken_actions[paraphrase_cnt]

        # Add regularizer maybe (reg_loss)
        pg_loss = tf.contrib.seq2seq.sequence_loss(
                logits=action_logit,
                targets=taken_action,
                weights=self.question_mask,
                average_across_batch=False)
        tf.summary.scalar('policy loss', tf.reduce_mean(pg_loss))
        pg_loss *= advantage
       
        # Optimize only paraphrase params 
        self.policy_params = [p for p in tf.trainable_variables()
                if (('Paraphrase_Layer' in p.name)
                or ('Similarity_Layer' in p.name))]
        print('pp optimize', [p.name for p in self.policy_params])

        # Regularizer not used
        reg_loss = tf.reduce_sum(
                [tf.reduce_sum(tf.square(x)) for x in self.policy_params]) * 0.001
        total_loss = tf.reduce_mean(pg_loss)
        
        self.policy_gradients = self.optimizer.compute_gradients(total_loss,
                var_list=self.policy_params)
        optimize = self.optimizer.apply_gradients(self.policy_gradients,
                global_step=self.global_step)

        self.pp_optimize.append(optimize)
        self.pp_loss.append(total_loss)
        
        for grad, var in self.policy_gradients:
            tf.summary.histogram(var.name, var)
            if grad is not None:
                tf.summary.histogram(var.name + '/gradients', grad)
        tf.summary.scalar('reg loss', reg_loss)
        tf.summary.scalar('total loss', tf.reduce_mean(total_loss))
        tf.summary.scalar('advantage', tf.reduce_mean(advantage))

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
            if pp_idx > 0: # Start paraphrase
                # Similarity calculate
                similarity_q = (question_embed if self.similarity_q == 'e'
                        else question_rep)
                similarity_c = (context_embed if self.similarity_c == 'e'
                        else context_rep)
                candidate = self.similarity_layer(similarity_c, similarity_q,
                        context_rep)

                # Policy network for paraphrase
                policy_q = (question_embed if self.policy_q == 'e'
                        else question_rep)
                _, action_logit = self.paraphrase_layer(
                        policy_q, c_state,
                        self.question_len, self.question_maxlen, 
                        candidate=candidate, reuse=(pp_idx>1))
                
                # Return policy and receive sample
                self.action_probs.append(tf.nn.softmax(
                    tf.cast(action_logit, dtype=tf.float64)))
                paraphrased = self.paraphrases[pp_idx-1]
                print('# Paraphrase_layer %d' % (pp_idx), action_logit)

            else: # No paraphrase (pp_idx = 0)
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
                        if ('Paraphrase_Layer' not in p.name) and
                        ('Similarity_Layer' not in p.name)]
                self.optimize_loss(
                        self.start_logits, self.end_logits, general_params)

    def anneal_exploration(self):
        if self.exploration > 0:
            self.exploration -= 0.1
        print('exploration annealed to %.1f' % self.exploration)

