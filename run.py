import tensorflow as tf
import numpy as np


def train(model, params, dataset):
    sess = model.session
    batch_size = params['batch_size']
    mini_batch = []
    for dataset_idx, dataset_item in enumerate(dataset):
        context = dataset_item['c']
        context_len = dataset_item['c_len']
        for qa in dataset_item['qa']:
            question = qa['q']
            question_len = qa['q_len']
            answer = qa['a']
            mini_batch.append([context, context_len, question, question_len, answer])
           
            # Run and clear mini-batch
            if (len(mini_batch) == batch_size) or (dataset_idx == len(dataset) - 1):
                batch_context = np.array([c for c, _, _, _, _ in mini_batch])
                batch_context_len = np.array([c_len for _, c_len, _, _, _ in mini_batch])
                batch_question = np.array([q for _, _, q, _, _ in mini_batch])
                batch_question_len = np.array([q_len for _, _, _, q_len, _ in mini_batch])
                batch_answer = np.array([a for _, _, _, _, a in mini_batch])

                """
                # Debugging
                print(batch_context.shape, batch_context_len.shape, 
                        batch_question.shape, batch_question_len.shape, batch_answer.shape)
                print('c', batch_context[0])
                print('c_len', batch_context_len[0])
                print('q', batch_question[0])
                print('q_len', batch_question_len[0])
                print('a', batch_answer[0])
                """

                feed_dict = {model.context: batch_context,
                        model.context_len: batch_context_len,
                        model.question: batch_question,
                        model.question_len: batch_question_len,
                        model.answer: batch_answer,
                        model.lstm_dropout: params['lstm_dropout'],
                        model.hidden_dropout: params['hidden_dropout']}
                _, loss = sess.run([model.optimize, model.loss], feed_dict=feed_dict)
                mini_batch = []

                # TODO: Evaluate f1 and em
                if dataset_idx % 5 == 0:
                    print("loss: %.3f, f1: %.3f, em: %.3f" % (loss, 0, 0))



def test(model, params, dataset):
    print("\nloss: %.3f, f1: %.3f, em: %.3f" % (0, 0, 0))

