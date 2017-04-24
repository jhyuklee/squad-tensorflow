import tensorflow as tf
import os

from ops import *


class Basic(object):
    def __init__(self, params, initializer):

        # session settings
        config = tf.ConfigProto(device_count={'GPU':1})
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        self.session = tf.Session(config=config)
        self.params = params
        self.model_name = params['model_name']

        # hyper parameters
        self.learning_rate = params['learning_rate']
        self.decay_rate = params['decay_rate']
        self.decay_step = params['decay_step']
        self.min_grad = params['min_grad']
        self.max_grad = params['max_grad']

        # rnn parameters
        self.context_maxlen = params['context_maxlen']
        self.question_maxlen = params['question_maxlen']
        self.cell_layer_num = params['lstm_layer']
        self.dim_word = params['dim_word'] 
        self.dim_embed_word = params['dim_embed_word']
        self.dim_hidden = params['dim_hidden']
        self.dim_rnn_cell = params['dim_rnn_cell']
        self.dim_output = params['dim_output']
        self.embed = params['embed']
        self.embed_trainable = params['embed_trainable']
        self.checkpoint_dir = params['checkpoint_dir']
        self.initializer = initializer

        # input data placeholders
        self.context = tf.placeholder(tf.int32, [None, self.context_maxlen])
        self.question = tf.placeholder(tf.int32, [None, self.question_maxlen])
        self.answer_start = tf.placeholder(tf.int32, [None])
        self.answer_end = tf.placeholder(tf.int32, [None])
        self.context_len = tf.placeholder(tf.int32, [None])
        self.question_len = tf.placeholder(tf.int32, [None])
        self.lstm_dropout = tf.placeholder(tf.float32)
        self.hidden_dropout = tf.placeholder(tf.float32)

        # model settings
        self.global_step = tf.Variable(0, name="step", trainable=False)
        self.learning_rate = tf.train.exponential_decay(
                self.learning_rate, self.global_step,
                self.decay_step, self.decay_rate, staircase=True)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.optimize = None
        self.saver = None
        self.loss = None
        self.start_logits = None
        self.end_logits = None

        # model build
        self.merged_summary = None
        self.embed_writer = tf.summary.FileWriter(self.checkpoint_dir)
        self.embed_config = projector.ProjectorConfig()
        self.projector = None 
        if params['embed_pretrained']:
           embeddings = self.initialize_embedding(initializer)
        self.build_model()
        self.optimize_loss(self.start_logits, self.end_logits)
        self.save_settings()
        self.session.run(tf.global_variables_initializer())
        if params['embed_pretrained']:
            self.session.run(embeddings)
       
        # debug initializer
        with tf.variable_scope('Word', reuse=True):
            variable_here = tf.get_variable("embed", [self.dim_word, self.dim_embed_word],
                    dtype=tf.float32)
            print(variable_here.eval(session=self.session))
        

    def encoder(self, inputs, length, max_length, dim_input, dim_embed, 
            initializer=None, trainable=True, reuse=False, scope='encoding'):
        
        inputs_embed = embedding_lookup(
                inputs, 
                dim_input, 
                dim_embed,
                initializer=initializer, trainable=trainable, 
                reuse=reuse, scope='Word')

        with tf.variable_scope(scope) as scope: 
            fw_cell = lstm_cell(self.dim_rnn_cell, self.cell_layer_num, self.lstm_dropout)
            bw_cell = lstm_cell(self.dim_rnn_cell, self.cell_layer_num, self.lstm_dropout) 
            inputs_reshape = rnn_reshape(inputs_embed, dim_embed, max_length)
            outputs = rnn_model(inputs_reshape, length, max_length, fw_cell, self.params)
            return outputs
    

    def build_model(self):
        print("###  Building a Basic model ###")
        context_encoded = self.encoder(inputs=self.context,
                length=self.context_len,
                max_length=self.context_maxlen,
                dim_input=self.dim_word,
                dim_embed=self.dim_embed_word,
                trainable=self.embed_trainable,
                reuse=True, scope='Context')

        question_encoded = self.encoder(inputs=self.question,
                length=self.question_len,
                max_length=self.question_maxlen,
                dim_input=self.dim_word,
                dim_embed=self.dim_embed_word,
                trainable=self.embed_trainable,
                reuse=True, scope='Question')

        cct = tf.concat(axis=1, values=[context_encoded, question_encoded])

        hidden1 = linear(inputs=cct,
                output_dim=self.dim_hidden,
                dropout_rate=self.hidden_dropout,
                activation=tf.nn.relu,
                scope='Hidden1')

        self.start_logits = linear(inputs=hidden1,
            output_dim=self.dim_output, 
            scope='Output_s')

        self.end_logits = linear(inputs=hidden1,
            output_dim=self.dim_output, 
            scope='Output_e')
    
    def optimize_loss(self, start_logits, end_logits):
        start_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.start_logits, labels=self.answer_start)) 
        end_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.end_logits, labels=self.answer_end))
        self.loss = start_loss + end_loss
        tf.summary.scalar('Loss', self.loss)

        grads = []
        self.variables = tf.trainable_variables()
        for grad in tf.gradients(self.loss, self.variables):
            if grad is not None:
                grads.append(tf.clip_by_value(grad, self.min_grad, self.max_grad))
            else:
                grads.append(grad)
        self.optimize = self.optimizer.apply_gradients(zip(grads, self.variables), 
                global_step=self.global_step)
    
    def initialize_embedding(self, word_embed):
        with tf.variable_scope("Word"):
            word_embeddings = tf.get_variable("embed",
                                              initializer=tf.constant(word_embed),
                                              trainable=self.embed_trainable,
                                              dtype=tf.float32)
            return word_embeddings

    def save_settings(self):
        model_vars = [v for v in tf.global_variables()]
        print('model variables', [model_var.name for model_var in tf.trainable_variables()])
        self.saver = tf.train.Saver(model_vars)
        self.merged_summary = tf.summary.merge_all()

    @staticmethod
    def reset_graph():
        tf.reset_default_graph()

    def save(self, checkpoint_dir, step):
        file_name = "%s.model" % self.model_name
        self.saver.save(self.session, os.path.join(checkpoint_dir, file_name))
        print("Model saved", file_name)

    def load(self, checkpoint_dir):
        file_name = "%s.model" % self.model_name
        file_name += "-10800"
        self.saver.restore(self.session, os.path.join(checkpoint_dir, file_name))
        print("Model loaded", file_name)

