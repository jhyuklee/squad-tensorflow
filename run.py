import tensorflow as tf
import numpy as np

from evaluate import *
from utils import *

def run_paraphrase(question, question_len, context_raws, context_len, ground_truths,
        sim_mat, baseline_em, baseline_f1, pp_idx, model, feed_dict, params, is_train):

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
            new_sentence.append(1) # PAD token

        dprint('\nOriginal %s'% sentence[:length], params['debug'])
        dprint('Rules %s'% 
                (' '.join([idx2action[idx] for idx in actions[:length]])),
                params['debug'])
        dprint('Paraphrase %s'% new_sentence[:length], params['debug'])
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
        feed_dict[model.rewards[pp_idx]] = [(em + f1) for em, f1 in zip(em_s, f1_s)]
        feed_dict[model.baselines[pp_idx]] = [baseline_f1 + baseline_em]
        _, pp_loss = sess.run([
            model.pp_optimize[pp_idx],
            model.pp_loss[pp_idx]], feed_dict=feed_dict)
    else:
        pp_loss = 0.0
    
    pp_em = np.sum(em_s) / len(question)
    pp_f1 = np.sum(f1_s) / len(question)

    return pp_em, pp_f1, pp_loss


def run_epoch(model, dataset, epoch, idx2word, params, is_train=True):
    print('### Training ###' if is_train else '\n### Testing ###')
    sess = model.session
    batch_size = params['batch_size']
    mini_batch = []
    ground_truths = []
    context_raws = []
    question_raws = []  # Not used with idx2word
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
            mini_batch.append([context, context_len, 
                question, question_len, answer_start, answer_end])
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
                
                # Use 1.0 dropout for test time
                if not is_train:
                    feed_dict[model.rnn_dropout] = 1.0
                    feed_dict[model.hidden_dropout] = 1.0
                    feed_dict[model.embed_dropout] = 1.0

                # do not train when 'pp_only'
                if not (params['mode'] == 'q' and params['train_pp_only']) and is_train:
                    sess.run(model.optimize, feed_dict=feed_dict)
                
                loss, start_logits, end_logits, lr = sess.run(
                        [model.loss, model.start_logits, model.end_logits, 
                            model.learning_rate], feed_dict=feed_dict)
                
                predictions = pred_from_logits(start_logits, 
                        end_logits, batch_context_len, context_raws, params)
                em, f1 = em_f1_score(predictions, ground_truths, params)

                # running_em = total_em / (total_cnt + 1e-5)
                # running_f1 = total_f1 / (total_cnt + 1e-5)
                baseline_em = em
                baseline_f1 = f1

                if 'q' == params['mode']:
                    for pp_idx in range(params['num_paraphrase']):
                        tmp_em, tmp_f1, tmp_loss = run_paraphrase(
                                batch_question,
                                batch_question_len,
                                context_raws,
                                batch_context_len,
                                ground_truths, None, # Use similarity matrix
                                baseline_em, baseline_f1, pp_idx,
                                model, feed_dict, params, is_train=is_train)
                        pp_em[pp_idx] += tmp_em
                        pp_f1[pp_idx] += tmp_f1
                        pp_losses[pp_idx] += tmp_loss
                        pp_cnt += 1
                
                # Print intermediate result
                if dataset_idx % 5 == 0:
                    em = np.sum(em) / len(mini_batch)
                    f1 = np.sum(f1) / len(mini_batch)
                    
                    _progress = progress(dataset_idx / float(len(dataset)))
                    _progress += "loss: %.3f, em: %.3f, f1: %.3f" % (loss, em, f1)
                    _progress += " progress: %d/%d, lr: %.5f, ep: %d" %(
                            dataset_idx, len(dataset), lr, epoch)
                    sys.stdout.write(_progress)
                    sys.stdout.flush()
                    
                    total_f1 += f1
                    total_em += em
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
    print('\nAverage loss: %.3f, f1: %.3f, em: %.3f' % (
        total_loss, total_f1, total_em))

    if 'q' in params['mode']:
        pp_losses[0] /= pp_cnt
        pp_em[0] /= pp_cnt
        pp_f1[0] /= pp_cnt
        print('Paraphrase loss: %.3f, f1: %.3f, em: %.3f' % (
            pp_losses[0], pp_f1[0], pp_em[0]))

    return total_em, total_f1, total_loss

