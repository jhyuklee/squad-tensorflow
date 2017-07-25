import tensorflow as tf
import numpy as np

from evaluate import *
from utils import *

def run_paraphrase(question, question_len, context_raws, context_len, 
        ground_truths, sim_mat, baseline_em, baseline_f1, pp_idx, idx2word, 
        model, feed_dict, params, is_train):

    idx2action = {
            0: 'NONE',
            1: 'DEL',
            2: 'SUB',
            3: 'INS'
    }
    sess = model.session
    action_prob, similarity = sess.run(
            [model.action_probs[pp_idx], model.selected_context],
            feed_dict=feed_dict)

    def softmax(logit):
        logit = np.exp(logit - np.amax(logit))
        logit = logit / np.sum(logit)
        return logit

    taken_action = []
    for batch_action in action_prob:
        actions = []
        for prob in batch_action:
            if np.random.random() < model.exploration:
                actions.append(np.random.randint(model.dim_action))
            else:
                actions.append(np.argmax(np.random.multinomial(1, prob)))
        taken_action.append(actions)

    def paraphrase_question(sentence, length, 
            actions, actions_prob, sim, max_action):
        new_sentence = []
        action_cnt = 0
        itr = 0

        # Choose max actions that is not None
        valid_probs = np.array([a[idx] for a, idx in zip(actions_prob, actions)])
        valid_probs *= np.array(actions).astype(bool)
        max_actions = valid_probs.argsort()[-max_action:][::-1]
        dprint([(m, idx2action[actions[m]]) for m in max_actions], params['debug'])

        for idx, act in enumerate(actions):
            if idx not in max_actions:
                new_sentence.append(sentence[itr])
                itr += 1
            elif idx2action[act] == 'NONE':
                new_sentence.append(sentence[itr])
                itr += 1
            elif idx2action[act] == 'DEL':
                itr += 1
            elif idx2action[act] == 'SUB':
                new_sentence.append(sim[itr])
                itr += 1
            elif idx2action[act] == 'INS':
                new_sentence.append(sentence[itr])
                new_sentence.append(sim[itr])
                itr += 1
            else:
                assert False, 'Invalid action %d'% act

            if itr >= length:
                break
       
        new_length = len(new_sentence)
        while len(new_sentence) <= len(sentence):
            new_sentence.append(1) # PAD token
        new_sentence = new_sentence[:model.question_maxlen]
        new_length = (new_length if new_length < model.question_maxlen
                else model.question_maxlen)

        return new_sentence, new_length
   
    # Get paraphrased question according to the taken_action (batch unpack)
    paraphrased_q = []
    paraphrased_qlen = []
    for org_q, org_q_len, action, a_prob, sim in zip(
            question, question_len, taken_action, action_prob, similarity):
        new_q, new_qlen = paraphrase_question(
                org_q, org_q_len, action, a_prob, sim, model.max_action)
        paraphrased_q.append(new_q)
        paraphrased_qlen.append(new_qlen)

    # Get scores for paraphrased question
    feed_dict[model.paraphrases[pp_idx]] = np.array(paraphrased_q)
    feed_dict[model.question_len] = np.array(paraphrased_qlen)
    ps_logits, pe_logits = sess.run(model.pp_logits[pp_idx], feed_dict=feed_dict)
    predictions = pred_from_logits(ps_logits, pe_logits,
            context_len, context_raws, params)
    em_s, f1_s = em_f1_score(predictions, ground_truths, params)

    dprint('\nparaphrased em %s' % em_s, params['debug'])
    dprint('baeline em %s' % baseline_em, params['debug'])
    dprint('advantage em %s' % (em_s - baseline_em), params['debug'])
    max_idx = np.argmax(em_s + f1_s - baseline_em - baseline_f1)
    dprint('max idx: %d, em: %.3f, f1: %.3f' % (
        max_idx, em_s[max_idx], f1_s[max_idx]), params['debug'])
    dprint('\nRules %s'% (' '.join([idx2action[idx]
                for idx in taken_action[max_idx][:question_len[max_idx]]])), 
                params['debug'])
    dprint('original %s' % [idx2word[w] 
        for w in question[max_idx][:question_len[max_idx]]], params['debug'])
    dprint('similarity %s' % [idx2word[w] 
        for w in similarity[max_idx][:question_len[max_idx]]], params['debug'])
    dprint('changed %s' % [idx2word[w] 
        for w in paraphrased_q[max_idx][:paraphrased_qlen[max_idx]]], params['debug'])
   
    # Use REINFORE with original em, f1 as baseline (per example)
    rewards = np.sum([em_s, f1_s], axis=0) / 2
    baselines = np.sum([baseline_em, baseline_f1], axis=0) / 2
    advantages = np.clip(
            rewards / (baselines + 1e-5) - 1, -1, params['rb_clip']) + rewards
    feed_dict[model.taken_actions[pp_idx]] = taken_action
    feed_dict[model.advantages[pp_idx]] = advantages
    _, pp_loss, summary = sess.run([
        model.pp_optimize[pp_idx] if is_train else model.no_op,
        model.pp_loss[pp_idx], model.merged_summary], feed_dict=feed_dict)

    advantages = np.mean(advantages)
    rewards = np.mean(rewards)
    baselines = np.mean(baselines)
    pp_em = np.sum(em_s) / len(question)
    pp_f1 = np.sum(f1_s) / len(question)

    return pp_em, pp_f1, pp_loss, advantages, rewards, baselines, summary


def run_epoch(model, dataset, epoch, base_iter, idx2word, params, is_train=True):
    print('### Training ###' if is_train else '\n### Testing ###')
    sess = model.session
    batch_size = params['batch_size']
    mini_batch = []
    ground_truths = []
    context_raws = []
    question_raws = []  # Not used 
    total_loss = total_f1 = total_em = total_cnt = 0
    pp_em = [0] * params['num_paraphrase']
    pp_f1 = [0] * params['num_paraphrase']
    pp_losses = [0] * params['num_paraphrase']
    pp_reward = [0] * params['num_paraphrase']
    pp_baseline = [0] * params['num_paraphrase']
    pp_advantage = [0] * params['num_paraphrase']
    pp_cnt = 0

    for dataset_idx, dataset_item in enumerate(dataset):
        context = dataset_item['c']
        context_raw = dataset_item['c_raw']
        context_len = dataset_item['c_len']
        for qa_idx, qa in enumerate(dataset_item['qa']):
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
            if (len(mini_batch) == batch_size) or ((dataset_idx == len(dataset) - 1) 
                    and (qa_idx == len(dataset_item['qa']) - 1)):
                batch_context = np.array([b[0] for b in mini_batch])
                batch_context_len = np.array([b[1] for b in mini_batch])
                batch_question = np.array([b[2] for b in mini_batch])
                batch_question_len = np.array([b[3] for b in mini_batch])
                batch_answer_start = np.array([b[4] for b in mini_batch])
                batch_answer_end = np.array([b[5] for b in mini_batch])

                # No dropout for question learning
                if params['mode'] == 'q':
                    params['rnn_dropout'] = 1.0
                    params['hidden_dropout'] = 1.0
                    params['embed_dropout'] = 1.0

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
                loss, start_logits, end_logits, lr, _ = sess.run(
                        [model.loss, model.start_logits, model.end_logits, 
                            model.learning_rate,
                            model.optimize if not (params['mode'] == 'q' 
                                and params['train_pp_only'])
                            and is_train else model.no_op], feed_dict=feed_dict)
                
                predictions = pred_from_logits(start_logits, 
                        end_logits, batch_context_len, context_raws, params)
                em, f1 = em_f1_score(predictions, ground_truths, params)

                baseline_em = em
                baseline_f1 = f1
                if 'q' == params['mode']:
                    for pp_idx in range(params['num_paraphrase']):
                        tmp_em, tmp_f1, tmp_loss, adv, tmp_r, tmp_b, summary = \
                                run_paraphrase(
                                        batch_question,
                                        batch_question_len,
                                        context_raws,
                                        batch_context_len,
                                        ground_truths, None, # Deprecated
                                        baseline_em, baseline_f1, pp_idx, idx2word,
                                        model, feed_dict, params, is_train=is_train)
                        pp_em[pp_idx] += tmp_em
                        pp_f1[pp_idx] += tmp_f1
                        pp_losses[pp_idx] += tmp_loss
                        pp_reward[pp_idx] += tmp_r
                        pp_baseline[pp_idx] += tmp_b
                        pp_advantage[pp_idx] += adv
                        pp_cnt += 1
                
                # Print intermediate result
                if dataset_idx % 5 == 0:
                    em = np.sum(em) / len(mini_batch)
                    f1 = np.sum(f1) / len(mini_batch)

                    if params['summarize'] and params['mode'] == 'q':
                        # Basic summary
                        summary_writer = (model.train_writer if is_train
                                else model.valid_writer)
                        summary_writer.add_summary(
                                summary, base_iter + pp_cnt)

                        # Cumulative summary
                        write_scalar_summary(
                                'cumulative reward',
                                pp_reward[0]/pp_cnt,
                                base_iter + pp_cnt,
                                summary_writer)
                        write_scalar_summary(
                                'cumulative baseline',
                                pp_baseline[0]/pp_cnt,
                                base_iter + pp_cnt,
                                summary_writer)
                        write_scalar_summary(
                                'cumulative advantage',
                                pp_advantage[0]/pp_cnt,
                                base_iter + pp_cnt,
                                summary_writer)

                    _progress = progress(dataset_idx / float(len(dataset)))
                    _progress += "loss:%.2f, em:%.2f, f1:%.2f" % (loss, em, f1)
                    _progress += ", idx:%d/%d [e%d]" %(
                            dataset_idx, len(dataset), epoch)
                    if 'q' == params['mode']:
                        _progress += " adv:%.2f" % (pp_advantage[0]/pp_cnt)
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
    total_em /= total_cnt
    total_f1 /= total_cnt
    total_loss /= total_cnt
    print('\nAverage loss: %.3f, em: %.3f, f1: %.3f' % (
        total_loss, total_em, total_f1))

    if 'q' in params['mode']:
        model.anneal_exploration()
        pp_em[0] /= pp_cnt
        pp_f1[0] /= pp_cnt
        pp_losses[0] /= pp_cnt
        pp_advantage[0] /= pp_cnt
        print('Paraphrase loss: %.3f, em: %.3f, f1: %.3f, adv: %.3f' % (
            pp_losses[0], pp_em[0], pp_f1[0], pp_advantage[0]))
    print('Total iteration %d' % (pp_cnt + base_iter))

    return total_em, total_f1, total_loss, pp_cnt + base_iter

