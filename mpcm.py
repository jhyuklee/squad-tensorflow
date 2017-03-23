import tensorflow as tf

from model import Basic
from ops import *


class MPCM(Basic):
    def __init__(self, params, initializer):
        super(MPCM, self).__init__(params, initializer)

    def filter_layer(self, context, question):
        c_norm = tf.norm(context, axis=2)
        q_norm = tf.norm(question, axis=2)
        print(c_norm, q_norm)
        
        # TODO: cosine similarity
        return 'filter_layer'

    def representation_layer(self, inputs, length, max_length):

        # TODO: bi-lstm
        return 'representation_layer'


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
        context_rep = self.representation_layer(context_filtered, self.context_len,
                self.context_maxlen)
        question_rep = self.representation_layer(question_embed, self.question_len,
                self.question_maxlen)

        

        
        




