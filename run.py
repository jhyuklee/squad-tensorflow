import tensorflow as tf
import numpy as np

from evaluate import *
from utils import *

def run_paraphrase(question, question_len, context, context_len, 
        context_raws, ground_truths, baseline_em, baseline_f1, lang_model, 
        pp_idx, idx2word, model, feed_dict, params, is_train):

    idx2action = { # 4=NDSI0, 6=NDSI1, 8=NDSI2, 11: NDSI2B
            0: 'NONE',
            1: 'DEL',
            2: 'SUB0',
            3: 'SUB1',
            4: 'SUB2',
            5: 'INS0F',
            6: 'INS1F',
            7: 'INS2F',
            8: 'INS0B',
            9: 'INS1B',
            10: 'INS2B'
    }
    sess = model.session
    action_prob, c_sim = sess.run(
            [model.action_probs[pp_idx], model.c_sim],
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
            actions, actions_prob, c_s, c_org, max_a):
        new_sentence = []
        action_cnt = 0
        itr = 0
        edit_distance = 0

        # Choose max actions that is not 'NONE'
        valid_probs = np.array([a[idx] for a, idx in zip(actions_prob, actions)])
        valid_probs *= np.array(actions).astype(bool)
        max_actions = valid_probs.argsort()[-max_a:][::-1]
        # dprint([(m, idx2action[actions[m]]) for m in max_actions], params['debug'])

        for idx, act in enumerate(actions):
            if (idx not in max_actions) and max_a > 0:
                print('not here please')
                new_sentence.append(sentence[itr])
                itr += 1
            elif idx2action[act] == 'NONE':
                new_sentence.append(sentence[itr])
                itr += 1
            elif idx2action[act] == 'DEL':
                itr += 1
                edit_distance += 1
            elif idx2action[act] == 'SUB0':
                new_sentence.append(c_org[c_s[itr]])
                itr += 1
                edit_distance += 1
            elif idx2action[act] == 'SUB1':
                new_sentence.append(c_org[c_s[itr]])
                if c_s[itr] < params['context_maxlen']-1:
                    new_sentence.append(c_org[c_s[itr]+1])
                itr += 1
                edit_distance += 2
            elif idx2action[act] == 'SUB2':
                new_sentence.append(c_org[c_s[itr]])
                if c_s[itr] < params['context_maxlen']-1:
                    new_sentence.append(c_org[c_s[itr]+1])
                if c_s[itr] < params['context_maxlen']-2:
                    new_sentence.append(c_org[c_s[itr]+2])
                itr += 1
                edit_distance += 3
            elif idx2action[act] == 'INS0F':
                new_sentence.append(sentence[itr])
                new_sentence.append(c_org[c_s[itr]])
                itr += 1
                edit_distance += 1
            elif idx2action[act] == 'INS1F':
                new_sentence.append(sentence[itr])
                new_sentence.append(c_org[c_s[itr]])
                if c_s[itr] < params['context_maxlen']-1:
                    new_sentence.append(c_org[c_s[itr]+1])
                itr += 1
                edit_distance += 2
            elif idx2action[act] == 'INS2F':
                new_sentence.append(sentence[itr])
                new_sentence.append(c_org[c_s[itr]])
                if c_s[itr] < params['context_maxlen']-1:
                    new_sentence.append(c_org[c_s[itr]+1])
                if c_s[itr] < params['context_maxlen']-2:
                    new_sentence.append(c_org[c_s[itr]+2])
                itr += 1
                edit_distance += 3
            elif idx2action[act] == 'INS0B':
                new_sentence.append(c_org[c_s[itr]])
                new_sentence.append(sentence[itr])
                itr += 1
                edit_distance += 1
            elif idx2action[act] == 'INS1B':
                new_sentence.append(c_org[c_s[itr]])
                if c_s[itr] < params['context_maxlen']-1:
                    new_sentence.append(c_org[c_s[itr]+1])
                new_sentence.append(sentence[itr])
                itr += 1
                edit_distance += 2
            elif idx2action[act] == 'INS2B':
                new_sentence.append(c_org[c_s[itr]])
                if c_s[itr] < params['context_maxlen']-1:
                    new_sentence.append(c_org[c_s[itr]+1])
                if c_s[itr] < params['context_maxlen']-2:
                    new_sentence.append(c_org[c_s[itr]+2])
                new_sentence.append(sentence[itr])
                itr += 1
                edit_distance += 3
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

        return new_sentence, new_length, edit_distance
   
    # Get paraphrased question according to the taken_action (batch unpack)
    paraphrased_q = []
    paraphrased_qlen = []
    edit_distances = []
    for org_q, org_q_len, action, a_prob, c_s, org_c in zip(
            question, question_len, taken_action, action_prob, c_sim, context):
        new_q, new_qlen, ed = paraphrase_question(
                org_q, org_q_len, action, a_prob, c_s, org_c, model.max_action)
        paraphrased_q.append(new_q)
        paraphrased_qlen.append(new_qlen)
        edit_distances.append(ed)

    # Get scores for paraphrased question
    feed_dict[model.paraphrases[pp_idx]] = np.array(paraphrased_q)
    feed_dict[model.question_len] = np.array(paraphrased_qlen)
    ps_logits, pe_logits = sess.run(model.pp_logits[pp_idx], feed_dict=feed_dict)
    predictions = pred_from_logits(ps_logits, pe_logits,
            context_len, context_raws, params)
    em_s, f1_s = em_f1_score(predictions, ground_truths, params)

    dprint('\nparaphrased em %s' % em_s, params['debug'])
    dprint('baeline em %s' % baseline_em, params['debug'])
    dprint('advantage em %s' % (em_s + f1_s - baseline_em - baseline_f1), 
            params['debug'])
    max_idx = np.argmax(em_s + f1_s - baseline_em - baseline_f1)
    dprint('edit distance %s' % edit_distances, params['debug'])
    dprint('max idx: %d, before em:%.3f, f1:%.3f // after em:%.3f, f1:%.3f' % (
        max_idx, baseline_em[max_idx], baseline_f1[max_idx], 
        em_s[max_idx], f1_s[max_idx]), params['debug'])
    dprint('\nRules %s'% (' '.join([idx2action[idx]
                for idx in taken_action[max_idx][:question_len[max_idx]]])), 
                params['debug'])
    dprint('Action prob %s'% [aa
        for aa in action_prob[max_idx][:question_len[max_idx]]], params['debug'])
    dprint('original %s' % [idx2word[w] 
        for w in question[max_idx][:question_len[max_idx]]], params['debug'])
    dprint('similarity %s' % [idx2word[context[max_idx, w]]
        for w in c_sim[max_idx][:question_len[max_idx]]], params['debug'])
    dprint('changed %s' % [idx2word[w] 
        for w in paraphrased_q[max_idx][:paraphrased_qlen[max_idx]]], params['debug'])
    dprint('Model number %s'% model.ymdhms, params['debug']) 
    dprint('-----------------------------------------------------------------------', 
            params['debug'])

    # Use REINFORE with original em, f1 as baseline (per example)
    rewards = np.sum([em_s, f1_s], axis=0) / 2
    baselines = np.sum([baseline_em, baseline_f1], axis=0) / 2
    advantages = (np.clip(
            rewards / (baselines + 1e-5) - 1, -1, params['rb_clip']) * 0.9
            + rewards * 0.1)
    """
    advantages = rewards - baselines
    advantages = []
    for r, b, ed in zip(rewards, baselines, edit_distances):
        edr = ed / 100.0
        if r > b:
            advantages.append(1)
        elif r == b:
            if ed > 0:
                advantages.append(0.5)
            else:
                advantages.append(0)
        elif r < b:
            advantages.append(-1)
        else:
            assert False, 'Invalid r, b'
    """
    smry_advs = []
    for r, b in zip(rewards, baselines):
        if r > b:
            smry_advs.append(1)
        elif r == b:
            smry_advs.append(0)
        elif r < b:
            smry_advs.append(-1)
        else:
            assert False, 'Invalid r, b'

    feed_dict[model.taken_actions[pp_idx]] = taken_action
    feed_dict[model.advantages[pp_idx]] = advantages
    _, pp_loss, summary = sess.run([
        model.pp_optimize[pp_idx] if is_train else model.no_op,
        model.pp_loss[pp_idx], model.merged_summary], feed_dict=feed_dict)

    advantages = np.mean(smry_advs)
    rewards = np.mean(rewards)
    baselines = np.mean(baselines)
    pp_em = np.sum(em_s) / len(question)
    pp_f1 = np.sum(f1_s) / len(question)

    return pp_em, pp_f1, pp_loss, advantages, rewards, baselines, summary


def run_epoch(model, dataset, epoch, base_iter, idx2word, params, 
        is_train=True, lang_model=None):
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
        c_char_raw = dataset_item['c_char']
        c_char = dataset_item['c_char_idx']
        c_char_len = dataset_item['char_len']


        for qa_idx, qa in enumerate(dataset_item['qa']):
            question = qa['q']
            question_len = qa['q_len']
            question_raw = qa['q_raw']
            answer = qa['a']
            answer_start = qa['a_start']
            answer_end = qa['a_end']
            
            q_char_raw = qa['q_char']
            q_char = qa['q_char_idx']
            q_char_len = qa['q_char_len']

            mini_batch.append([context, context_len, 
                question, question_len, answer_start, 
                answer_end, c_char, c_char_len, q_char, q_char_len])
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
                batch_context_char = np.array([b[6] for b in mini_batch])
                batch_context_char_len = np.array([b[7] for b in mini_batch])
                batch_question_char = np.array([b[8] for b in mini_batch])
                batch_question_char_len = np.array([b[9] for b in mini_batch])

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
                        model.learning_rate: params['learning_rate'],
                        model.context_char: batch_context_char,
                        model.question_char : batch_question_char,
                        model.cnn_keep_prob : params['cnn_keep_prob']}
                
                # Use 1.0 dropout for test time
                if not is_train:
                    feed_dict[model.rnn_dropout] = 1.0
                    feed_dict[model.hidden_dropout] = 1.0
                    feed_dict[model.embed_dropout] = 1.0
                    feed_dict[model.cnn_keep_prob] = 1.0
                
                if params['mode'] == 'bidaf':
                    feed_dict[model.is_train] = is_train

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
                if params['mode'] == 'q':
                    for pp_idx in range(params['num_paraphrase']):
                        tmp_em, tmp_f1, tmp_loss, adv, tmp_r, tmp_b, summary = \
                                run_paraphrase(
                                        batch_question, batch_question_len,
                                        batch_context, batch_context_len,
                                        context_raws, ground_truths, 
                                        baseline_em, baseline_f1, lang_model,
                                        pp_idx, idx2word,
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
                    _progress += "loss:%.3f, em:%.3f, f1:%.3f" % (loss, em, f1)
                    _progress += ", idx:%d/%d [e%d]" %(
                            dataset_idx, len(dataset), epoch)
                    if params['mode'] == 'q':
                        _progress += " adv:%.3f" % (pp_advantage[0]/pp_cnt)
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

    if params['mode'] == 'q':
        if params['anneal_exp']:
            model.anneal_exploration()
        pp_em[0] /= pp_cnt
        pp_f1[0] /= pp_cnt
        pp_losses[0] /= pp_cnt
        pp_advantage[0] /= pp_cnt
        print('Paraphrase loss: %.3f, em: %.3f, f1: %.3f, adv: %.3f' % (
            pp_losses[0], pp_em[0], pp_f1[0], pp_advantage[0]))
    print('Total iteration %d' % (pp_cnt + base_iter))

    return total_em, total_f1, total_loss, pp_cnt + base_iter

