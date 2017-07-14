import tensorflow as tf
import numpy as np

from evaluate import *
from utils import *

def run_paraphrase(question, question_len, context_raws, context_len, ground_truths,
        running_em, running_f1, pp_idx, model, feed_dict, params, is_train):

    sess = model.session
    action_sample = sess.run(
            model.action_samples[pp_idx],
            feed_dict=feed_dict)
    idx2action = {
            0: 'NONE',
            1: 'DEL',
            2: 'INS',
            3: 'SUB'
    }

    def paraphrase_question(sentence, length, actions):
        new_sentence = []
        itr = 0
        for idx, act in enumerate(actions):
            if act == 0: # None
                new_sentence.append(sentence[itr])
                itr += 1
            elif act == 1: # DEL
                itr += 1
            elif act == 2: # INS, TODO: match context
                new_sentence.append(sentence[itr])
                itr += 1
            elif act == 3: # SUB, TODO: match context
                new_sentence.append(sentence[itr])
                itr += 1
            else:
                assert False, 'Wrong action %d'% act

            if itr >= length:
                break
        
        while len(new_sentence) != len(sentence):
            new_sentence.append(1) # PAD

        # print('Original', sentence)
        # print('Rules', ' '.join(
        #     [idx2action[idx] for idx in actions]))
        # print('Paraph', new_sentence)
        return new_sentence
    
    paraphrased_q = []
    for org_q, org_q_len, action in zip(question, question_len, action_sample):
        paraphrased_q.append(paraphrase_question(org_q, org_q_len, action))
    feed_dict[model.paraphrases[pp_idx]] = np.array(paraphrased_q)
                        
    ps_logits, pe_logits = sess.run(model.pp_logits[pp_idx], feed_dict=feed_dict)

    predictions = pred_from_logits(ps_logits, pe_logits,
            context_len, context_raws, params)
    em_s, f1_s = em_f1_score(predictions, ground_truths, params)
    
    if is_train:
        feed_dict[model.rewards[pp_idx]] = [(em + f1)
                for em, f1 in zip(em_s, f1_s)]
        feed_dict[model.baselines[pp_idx]] = [running_f1 + running_em]
        _, pp_loss = sess.run([
            model.pp_optimize[pp_idx],
            model.pp_loss[pp_idx]], feed_dict=feed_dict)
    else:
        pp_loss = 0.0
    
    pp_em = np.sum(em_s)
    pp_f1 = np.sum(f1_s)

    return pp_loss, pp_f1, pp_em 


def train(model, dataset, epoch, idx2word, params):
    print('### Training ###')
    sess = model.session
    batch_size = params['batch_size']
    mini_batch = []
    ground_truths = []
    context_raws = []
    question_raws = []
    g_norm_list = []
    total_loss = total_f1 = total_em = total_cnt = 0
    pp_losses = [0] * params['num_paraphrase']
    pp_f1 = [0] * params['num_paraphrase']
    pp_em = [0] * params['num_paraphrase']
    pp_cnt = 0

    for dataset_idx, dataset_item in enumerate(dataset):
        context = dataset_item['c']
        context_raw = dataset_item['c_raw']
        context_len = dataset_item['c_len']
        for qa in dataset_item['qa']:
            question = qa['q']
            question_len = qa['q_len']
            question_raw = qa['q_raw']
            answer = qa['a']
            answer_start = qa['a_start']
            answer_end = qa['a_end']
            mini_batch.append([context, context_len, question, 
                question_len, answer_start, answer_end])
            ground_truths.append(answer)
            context_raws.append(context_raw)
            question_raws.append(question_raw)
           
            # Run and clear mini-batch
            if (len(mini_batch) == batch_size) or (dataset_idx == len(dataset) - 1):
                batch_context = np.array([b[0] for b in mini_batch])
                batch_context_len = np.array([b[1] for b in mini_batch])
                batch_question = np.array([b[2] for b in mini_batch])
                batch_question_len = np.array([b[3] for b in mini_batch])
                batch_answer_start = np.array([b[4] for b in mini_batch])
                batch_answer_end = np.array([b[5] for b in mini_batch])

                feed_dict = {model.context: batch_context,
                        model.context_len: batch_context_len,
                        model.question: batch_question,
                        model.question_len: batch_question_len,
                        model.answer_start: batch_answer_start,
                        model.answer_end: batch_answer_end,
                        model.rnn_dropout: params['rnn_dropout'],
                        model.hidden_dropout: params['hidden_dropout'],
                        model.embed_dropout: params['embed_dropout'],
                        model.learning_rate: params['learning_rate']}

                if not params['train_pp_only']:
                    _, loss = sess.run([model.optimize, model.loss], feed_dict=feed_dict)

                running_em = total_em / (total_cnt + 1e-5)
                running_f1 = total_f1 / (total_cnt + 1e-5)

                if 'q' in params['mode']:
                    for pp_idx in range(params['num_paraphrase']):
                        loss, em, f1 = run_paraphrase(
                                batch_question,
                                batch_question_len,
                                context_raws,
                                batch_context_len,
                                ground_truths,
                                running_em, running_f1, pp_idx,
                                model, feed_dict, params, is_train=True)
                        pp_losses[pp_idx] += loss
                        pp_em[pp_idx] += em / len(mini_batch)
                        pp_f1[pp_idx] += f1 / len(mini_batch)
                        pp_cnt += 1
                
                # Print intermediate result
                if dataset_idx % 5 == 0:
                    grads, start_logits, end_logits, lr = sess.run(
                            [model.grads, model.start_logits, model.end_logits, 
                                model.learning_rate], feed_dict=feed_dict)
                    
                    predictions = pred_from_logits(start_logits, 
                            end_logits, batch_context_len, context_raws, params)
                    em, f1 = em_f1_score(predictions, ground_truths, params)
                    em = np.sum(em)
                    f1 = np.sum(f1)
                    
                    _progress = progress(dataset_idx / float(len(dataset)))
                    _progress += ("loss: %.3f, f1: %.3f, em: %.3f, progress: %d/%d, lr: %.5f, ep: %d" %
                            (loss, f1 / len(predictions), em / len(predictions), 
                            dataset_idx, len(dataset), lr, epoch))
                    sys.stdout.write(_progress)
                    sys.stdout.flush()
                    
                    total_f1 += f1 / len(predictions)
                    total_em += em / len(predictions)
                    total_loss += loss
                    total_cnt += 1
                    
                mini_batch = []
                ground_truths = []
                context_raws = []
                question_raws = []

    # Average result
    total_f1 /= total_cnt
    total_em /= total_cnt
    total_loss /= total_cnt
    print('\nAverage loss: %.3f, f1: %.3f, em: %.3f' % (total_loss, total_f1, total_em))

    if 'q' in params['mode']:
        pp_losses[0] /= pp_cnt
        pp_em[0] /= pp_cnt
        pp_f1[0] /= pp_cnt
        print('Paraphrase loss: %.3f, f1: %.3f, em: %.3f' % (pp_losses[0], pp_f1[0], pp_em[0]))


def test(model, dataset, params):
    print('\n### Testing ###')
    sess = model.session
    batch_size = params['batch_size']
    mini_batch = []
    ground_truths = []
    context_raws = []
    question_raws = []
    total_loss = total_f1 = total_em = total_cnt = 0
    pp_losses = [0] * params['num_paraphrase']
    pp_f1 = [0] * params['num_paraphrase']
    pp_em = [0] * params['num_paraphrase']
    pp_cnt = 0
    test_writer = open('./result/analysis.txt', 'w')

    for dataset_idx, dataset_item in enumerate(dataset):
        context = dataset_item['c']
        context_raw = dataset_item['c_raw']
        context_len = dataset_item['c_len']
        for qa in dataset_item['qa']:
            question = qa['q']
            question_len = qa['q_len']
            question_raw = qa['q_raw']
            answer = qa['a']
            answer_start = qa['a_start']
            answer_end = qa['a_end']
            mini_batch.append([context, context_len, question, question_len, answer_start,
                answer_end])
            ground_truths.append(answer)
            context_raws.append(context_raw)
            question_raws.append(question_raw)
           
            # Run and clear mini-batch
            if (len(mini_batch) == batch_size) or (dataset_idx == len(dataset) - 1):
                batch_context = np.array([b[0] for b in mini_batch])
                batch_context_len = np.array([b[1] for b in mini_batch])
                batch_question = np.array([b[2] for b in mini_batch])
                batch_question_len = np.array([b[3] for b in mini_batch])
                batch_answer_start = np.array([b[4] for b in mini_batch])
                batch_answer_end = np.array([b[5] for b in mini_batch])

                feed_dict = {model.context: batch_context,
                        model.context_len: batch_context_len,
                        model.question: batch_question,
                        model.question_len: batch_question_len,
                        model.answer_start: batch_answer_start,
                        model.answer_end: batch_answer_end,
                        model.rnn_dropout: 1.0,
                        model.hidden_dropout: 1.0,
                        model.embed_dropout: 1.0}
                
                running_em = total_em / (total_cnt + 1e-5)
                running_f1 = total_f1 / (total_cnt + 1e-5)

                if 'q' in params['mode']:
                    for pp_idx in range(params['num_paraphrase']):
                        loss, em, f1 = run_paraphrase(
                                batch_question,
                                batch_question_len,
                                context_raws,
                                batch_context_len,
                                ground_truths,
                                running_em, running_f1, pp_idx,
                                model, feed_dict, params, is_train=False)
                        pp_losses[pp_idx] += loss
                        pp_em[pp_idx] += em / len(mini_batch)
                        pp_f1[pp_idx] += f1 / len(mini_batch)
                        pp_cnt += 1
                
                # Print intermediate result
                loss, start_logits, end_logits = sess.run(
                        [model.loss, model.start_logits, model.end_logits], 
                        feed_dict=feed_dict)
                start_idx = [np.argmax(sl[:cl], 0) 
                        for sl, cl in zip(start_logits, batch_context_len)]
                end_idx = [np.argmax(el[si:cl], 0) + si
                        for el, si, cl in zip(end_logits, start_idx, batch_context_len)]
                predictions = []

                for c, s_idx, e_idx in zip(context_raws, start_idx, end_idx):
                    predictions.append(' '.join([w for w in c[s_idx: e_idx+1]]))

                em = f1 = 0 
                for prediction, ground_truth, ctr, qur in zip(
                        predictions, ground_truths, context_raws, question_raws):
                    single_em = metric_max_over_ground_truths(
                            exact_match_score, prediction, ground_truth)
                    single_f1 = metric_max_over_ground_truths(
                            f1_score, prediction, ground_truth)

                    test_writer.write(('[Correct]' if single_f1 > 0 else '[Incorrect]') + '\n')
                    test_writer.write('C: ' + str(' '.join(ctr)) + '\n')
                    test_writer.write('Q: ' + str(' '.join(qur)) + '\n')
                    test_writer.write('ground_truth: ' + str(ground_truth) + '\n')
                    test_writer.write('prediction: ' + str(prediction) + '\n\n')

                    em += single_em
                    f1 += single_f1
                
                _progress = progress(dataset_idx / float(len(dataset)))
                _progress += "loss: %.3f, f1: %.3f, em: %.3f, progress: %d/%d" % (loss, 
                        f1 / len(predictions), 
                        em / len(predictions), dataset_idx, len(dataset)) 
                sys.stdout.write(_progress)
                sys.stdout.flush()

                total_f1 += f1 / len(predictions)
                total_em += em / len(predictions)
                total_loss += loss
                total_cnt += 1
                    
                mini_batch = []
                ground_truths = []
                context_raws = []
                question_raws = []
    
    test_writer.close()

    # Average result
    total_f1 /= total_cnt
    total_em /= total_cnt
    total_loss /= total_cnt
    print('\nAverage loss: %.3f, f1: %.3f, em: %.3f' % (total_loss, total_f1, total_em))
    
    if 'q' in params['mode']:
        pp_losses[0] /= pp_cnt
        pp_em[0] /= pp_cnt
        pp_f1[0] /= pp_cnt
        print('Paraphrase loss: %.3f, f1: %.3f, em: %.3f' % (pp_losses[0], pp_f1[0], pp_em[0]))

    return total_f1, total_em, total_loss

