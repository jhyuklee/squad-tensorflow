import tensorflow as tf
import os
import datetime

from ops import *


class Basic(object):
    def __init__(self, params, initializer):

        # session settings
        config = tf.ConfigProto(
                allow_soft_placement = True,
                # log_device_placement = True,
                device_count={'GPU':2}
        )
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)
        self.params = params
        self.model_name = params['model_name']
        self.ymdhms = params['ymdhms']

        # rnn parameters
        self.max_grad_norm = params['max_grad_norm']
        self.context_maxlen = params['context_maxlen']
        self.question_maxlen = params['question_maxlen']
        self.rnn_layer = params['rnn_layer']
        self.voca_size = params['voca_size']
        self.dim_embed_word = params['dim_embed_word']
        self.dim_hidden = params['dim_hidden']
        self.dim_rnn_cell = params['dim_rnn_cell']
        self.dim_output = params['dim_output']
        self.embed_trainable = params['embed_trainable']
        self.checkpoint_dir = params['checkpoint_dir']
        self.summary_dir = params['summary_dir']
        self.initializer, self.dictionary = initializer
        
        # character embedding parameters
        self.word_maxlen = params['word_maxlen']
        self.char_emb_dim = params['char_emb_dim']
        self.filter_width = params['filter_width']
        self.char_out = params['char_out']
        self.char_size = params['char_size']
        self.share_conv = params['share_conv']

        # input data placeholders
        self.context = tf.placeholder(tf.int32, [None, self.context_maxlen])
        self.question = tf.placeholder(tf.int32, [None, self.question_maxlen])
        self.answer_start = tf.placeholder(tf.int32, [None])
        self.answer_end = tf.placeholder(tf.int32, [None])
        self.context_len = tf.placeholder(tf.int32, [None])
        self.question_len = tf.placeholder(tf.int32, [None])
        self.rnn_dropout = tf.placeholder(tf.float32)
        self.hidden_dropout = tf.placeholder(tf.float32)
        self.embed_dropout = tf.placeholder(tf.float32)
        self.learning_rate = tf.placeholder(tf.float32)
        self.context_char = tf.placeholder(tf.int32, 
                [None, self.context_maxlen, self.word_maxlen])
        self.question_char = tf.placeholder(tf.int32, 
                [None, self.question_maxlen, self.word_maxlen])
        self.cnn_keep_prob = tf.placeholder(tf.float32)

        # model settings
        self.context_mask = tf.sequence_mask(self.context_len, 
                self.context_maxlen, dtype=tf.float32)
        self.question_mask = tf.sequence_mask(self.question_len, 
                self.question_maxlen, dtype=tf.float32)
        self.global_step = tf.Variable(0, name="step", trainable=False)
        if params['optimizer'] == 's': 
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        elif params['optimizer'] == 'm':
            self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.9)
        elif params['optimizer'] == 'a':
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        else:
            assert False, 'Wrong optimizer %s' % params['optimizer']
        self.no_op = tf.no_op()
        self.initialize_embedding(self.initializer)

        # build model
        start_time = datetime.datetime.now()
        self.build_model()
        self.save_settings()
        self.session.run(tf.global_variables_initializer())
        elapsed_time = datetime.datetime.now() - start_time
        print('Model Building Done', elapsed_time)
       
        # debug initializer
        with tf.variable_scope('Word', reuse=True):
            vv = tf.get_variable("embed", [self.voca_size, self.dim_embed_word],
                    dtype=tf.float32)
            print('apple:', self.dictionary['apple'])
            print(vv.eval(session=self.session)[self.dictionary['apple']][:5])
            print(vv.eval(session=self.session), '\n')
        
    def encoder(self, inputs, length, max_length, dim_input, dim_embed, 
            initializer=None, trainable=True, reuse=False, scope='encoding'):
        
        inputs_embed = dropout(embedding_lookup(
                inputs, 
                dim_input, 
                dim_embed,
                initializer=initializer, trainable=trainable, 
                reuse=reuse, scope='Word'), self.embed_dropout)

        with tf.variable_scope(scope) as scope: 
            fw_cell = lstm_cell(self.dim_rnn_cell, self.rnn_layer, self.rnn_dropout)
            bw_cell = lstm_cell(self.dim_rnn_cell, self.rnn_layer, self.rnn_dropout) 
            inputs_reshape = rnn_reshape(inputs_embed, dim_embed, max_length)
            outputs = rnn_model(
                    inputs_reshape, length, max_length, fw_cell, self.params)
            return outputs

    def build_model(self):
        print("###  Building a Basic model ###")
        context_encoded = self.encoder(inputs=self.context,
                length=self.context_len,
                max_length=self.context_maxlen,
                dim_input=self.voca_size,
                dim_embed=self.dim_embed_word,
                trainable=self.embed_trainable,
                reuse=True, scope='Context')

        question_encoded = self.encoder(inputs=self.question,
                length=self.question_len,
                max_length=self.question_maxlen,
                dim_input=self.voca_size,
                dim_embed=self.dim_embed_word,
                trainable=self.embed_trainable,
                reuse=True, scope='Question')

        cct = tf.concat(axis=1, values=[context_encoded, question_encoded])
        print('cct', cct)

        hidden1 = linear(inputs=cct,
                output_dim=self.dim_hidden,
                dropout_rate=self.hidden_dropout,
                activation=tf.nn.relu,
                scope='Hidden1')
        print('hidden', hidden1)

        start_logits = linear(inputs=hidden1,
            output_dim=self.dim_output,
            scope='Output_s')

        end_logits = linear(inputs=hidden1,
            output_dim=self.dim_output, 
            scope='Output_e')

        print('start, end logits', start_logits, end_logits)
        self.optimize_loss(start_logits, end_logits)
    
    def optimize_loss(self, start_logits, end_logits, vars=None):
        start_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=start_logits, labels=self.answer_start)) 
        end_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=end_logits, labels=self.answer_end))
        self.loss = start_loss + end_loss
        tf.summary.scalar('Loss', self.loss)
        
        print('# Calculating derivatives.. \n')
        if vars == None:
            self.variables = tf.trainable_variables()
        else:
            self.variables = vars
        self.grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, self.variables),
                self.max_grad_norm)
        self.optimize = self.optimizer.apply_gradients(
                zip(self.grads, self.variables), global_step=self.global_step)
    
    def initialize_embedding(self, word_embed):
        with tf.variable_scope("Word"):
            word_embeddings = tf.get_variable("embed",
                    initializer=tf.constant(word_embed),
                    trainable=self.embed_trainable,
                    dtype=tf.float32)

    def apply_mask(self, target, mask):
        if len(target.get_shape()) > len(mask.get_shape()):
            while len(mask.get_shape()) != len(target.get_shape()):
                mask = tf.expand_dims(mask, -1)
        return tf.multiply(target, mask)

    def save_settings(self):
        print('trainable variables', [var.name for var in tf.trainable_variables()])
        total_parameters = 0
        for var in tf.trainable_variables():
            shape = var.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            print(var.name, 'has %d parameters with %s shape' % (
                variable_parameters, shape))
            total_parameters += variable_parameters
        print('Total parameters', total_parameters)
        
        if self.params['mode'] == 'q':
            model_vars = [v for v in tf.trainable_variables()
                    if ('Paraphrase_Layer' not in v.name) and
                    ('Similarity_Layer' not in v.name)]
        else:
            model_vars = [v for v in tf.trainable_variables()]
        self.loader = tf.train.Saver(model_vars)
        model_vars = [v for v in tf.trainable_variables()]
        self.saver = tf.train.Saver(model_vars)
        self.merged_summary = tf.summary.merge_all()
        # could add self.session.graph
        if self.params['summarize']:
            self.train_writer = tf.summary.FileWriter(
                    self.summary_dir + self.ymdhms + '/train') 
            self.valid_writer = tf.summary.FileWriter(
                    self.summary_dir + self.ymdhms + '/valid') 

    @staticmethod
    def reset_graph():
        tf.reset_default_graph()

    def save(self, checkpoint_dir):
        file_name = "%s" % self.model_name
        self.saver.save(self.session, os.path.join(checkpoint_dir, file_name))
        print("Model saved", file_name)

    def load(self, checkpoint_dir):
        file_name = "%s" % self.model_name
        self.loader.restore(self.session, os.path.join(checkpoint_dir, file_name))
        print("Model loaded", file_name)

