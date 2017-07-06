import tensorflow as tf
import numpy as np

from evaluate import *
from utils import *


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
                _, loss = sess.run([model.optimize, model.loss], feed_dict=feed_dict)

                if 'q' in params['model']:
                    for pp_idx in range(params['num_paraphrase']):
                        ps_logits, pe_logits = sess.run(
                                model.paraphrase_logits[pp_idx],
                                feed_dict=feed_dict)
                        predictions = pred_from_logits(ps_logits, pe_logits,
                                batch_context_len, context_raws, params)
                        em_s, f1_s = em_f1_score(predictions, ground_truths, params)

                        running_f1 = total_f1 / (total_cnt + 1e-5)
                        running_em = total_em / (total_cnt + 1e-5)
                        feed_dict[model.rewards[pp_idx]] = [(em + f1)
                                for em, f1 in zip(em_s, f1_s)]
                        feed_dict[model.baselines[pp_idx]] = [running_f1 + running_em]
                        _, pp_sample, pp_loss = sess.run([
                            model.paraphrase_optimize[pp_idx],
                            model.paraphrases[pp_idx], model.policy_loss], 
                            feed_dict=feed_dict)

                        if dataset_idx % 5 == 0:
                            print()
                            for sample, q_raw in zip(pp_sample, question_raws):
                                pp = ' '.join([idx2word[idx] 
                                    for idx in sample[:len(q_raw)]])
                                qq = ' '.join(q_raw)
                                print('Sampled question: [%s]' % (pp))
                                print('Original question: [%s]' % (qq))
                            print('Paraphrased f1: %.3f, em: %.3f loss: %.3f' % (
                                np.sum(f1_s) / len(predictions), 
                                np.sum(em_s) / len(predictions),
                                pp_loss))
                
                # Print intermediate result
                if dataset_idx % 3 == 0:
                    """
                    # Dataset Debugging
                    print(batch_context.shape, batch_context_len.shape, 
                            batch_question.shape, batch_question_len.shape, 
                            batch_answer_start.shape)
                    for kk in range(len(batch_context)):
                        print('c', batch_context[kk][:10])
                        print('c_len', batch_context_len[kk])
                        print('q', batch_question[kk][:10])
                        print('q_len', batch_question_len[kk])
                        print('a', batch_answer_start[kk])
                        print('a', batch_answer_end[kk])
                    """
                
                    grads, start_logits, end_logits, lr = sess.run(
                            [model.grads, model.start_logits, model.end_logits, 
                                model.learning_rate], feed_dict=feed_dict)

                    predictions = pred_from_logits(start_logits, 
                            end_logits, batch_context_len, context_raws, params)

                    """
                    dprint('shape of grad/sl/el = %s/%s/%s' % (
                        np.asarray(grads).shape, np.asarray(start_logits).shape, 
                                 np.asarray(end_logits).shape), params['debug'])
                    g_norm_group = []
                    for gs in grads:
                        np_gs = np.asarray(gs)
                        g_norm = np.linalg.norm(np_gs)

                        norm_size = np_gs.shape
                        g_norm_group.append(g_norm)
                        dprint('g:' + str(g_norm) + str(norm_size), 
                                params['debug'], end=' ')
                    dprint('', params['debug'])
                    g_norm_list.append(g_norm_group)
                    """

                    for sl, el in zip(start_logits, end_logits):
                        # dprint('s:' + str(sl[:10]), params['debug'])
                        # dprint('e:' + str(el[:10]), params['debug'])
                        pass

                    em, f1 = em_f1_score(predictions, ground_truths, params)
                    em = np.sum(em)
                    f1 = np.sum(f1)
                    dprint('', params['debug'])
                    
                    _progress = progress(dataset_idx / float(len(dataset)))
                    _progress += ("loss: %.3f, f1: %.3f, em: %.3f, progress: %d/%d, lr: %.5f, ep: %d" %
                            (loss, f1 / len(predictions), em / len(predictions), 
                            dataset_idx, len(dataset), lr, epoch))
                    sys.stdout.write(_progress)
                    sys.stdout.flush()
                    
                    if dataset_idx / 5 == 5 and params['test']:
                        sys.exit()

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

    # Write norm information
    if params['debug']:
        f = open('./result/norm_info.txt', 'a')
        f.write('Norm Info\n')
        for g_norm_group in g_norm_list:
            s = '\t'.join([str(g) for g in g_norm_group]) + '\n'
            f.write(s)
        f.close()
    # sys.exit()


def test(model, dataset, params):
    print('\n### Testing ###')
    sess = model.session
    batch_size = params['batch_size']
    mini_batch = []
    ground_truths = []
    context_raws = []
    question_raws = []
    total_loss = total_f1 = total_em = total_cnt = 0
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
                
                # Print intermediate result
                loss, start_logits, end_logits = sess.run(
                        [model.loss, model.start_logits, model.end_logits], 
                        feed_dict=feed_dict)
                start_idx = [np.argmax(sl[:cl], 0) 
                        for sl, cl in zip(start_logits, batch_context_len)]
                end_idx = [np.argmax(el[si:cl], 0) + si
                        for el, si, cl in zip(end_logits, start_idx, batch_context_len)]
                predictions = []

                dprint('', params['debug'])
                for c, s_idx, e_idx in zip(context_raws, start_idx, end_idx):
                    dprint('si/ei=(%d/%d)'% (s_idx, e_idx), params['debug'], end= '\t')
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

                    prediction = prediction.split(' ') 
                    prediction = prediction[:10] if len(prediction) > 10 else prediction
                    dprint('pred: ' + str(' '.join(prediction)), 
                            params['debug'] and (single_f1 > 0))
                    dprint('real: ' + str(ground_truth), 
                            params['debug'] and (single_f1 > 0))

                    em += single_em
                    f1 += single_f1
                
                dprint('', params['debug'])
                
                _progress = progress(dataset_idx / float(len(dataset)))
                _progress += "loss: %.3f, f1: %.3f, em: %.3f, progress: %d/%d" % (loss, f1 /
                        len(predictions), em / len(predictions), dataset_idx, len(dataset)) 
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

    return total_f1, total_em, total_loss

