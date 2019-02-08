import sys; sys.path.append('/home/lgy/drl_cache')
from Agents import LRUAgent, LFUAgent, RLAgent
import pickle as cp
import tensorflow as tf
from utils.util import get_config_from_json, choose_group
from os import listdir
import random

try:
    config_file = "../configs/config_RL.json"
    config, _ = get_config_from_json(config_file)
except Exception as e:
    print("missing or invalid arguments %s" % e)
    exit(0)

def main():

    data_dir = config.reqs_dir_real
    target_group = choose_group(data_dir)

    with tf.Session() as sess:

        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        rlagent = RLAgent.RLagent(sess, config)
        lruagent = LRUAgent.LRUAgent(config)
        lfuagent = LFUAgent.LFUAgent(config)
        sess.run(tf.global_variables_initializer())
        totalreq = 0

        with open(config.LOG_FILE, 'wb') as log_file:
            for f in sorted(listdir(data_dir)): # use only 1 file
                with open(data_dir + f, 'rb') as infile:
                    reqlist = cp.load(infile)

                # for req in reqlist:
                while(True): # train forever
                    seed = random.randint(0, len(reqlist)-rlagent.TRAIN_SEQ_LEN-10)
                    req_samples = reqlist[seed : seed + rlagent.TRAIN_SEQ_LEN]

                    for req in req_samples:
                        # focus on only one target group
                        if req[1][:config.PREFIX] != target_group:
                            continue
                        rlagent.update_cache(req)
                        lruagent.update_cache(req)
                        lfuagent.update_cache(req)
                        totalreq += 1

                        if totalreq % 1000 == 0: # print result every xxx requests
                            config_str = "LRU_feat = y, " \
                                         "LFU_feat = n, " \
                                         "entropyW = 5, " \
                                         "normalized_data = y, " \
                                         "similar_acts = n, " \
                                         "activation = tanh, " \
                                         "train_len = 1000, " \
                                         "batch_apply = 2, " \
                                         "batch_norm = y, " \
                                         "random sample = y, " \
                                         "weight_init = xavier"
                            print(config_str)
                            print('CACHE_SIZE=', config.CACHE_SIZE,'WINDOW_SIZES=', config.WINDOW_SIZES,'PREFIX=', config.PREFIX)
                            print('totalreq=', totalreq)
                            print('RL_hit - LRU_hit:', rlagent.hit - lruagent.hit)
                            print('RL_hit - LFU_hit:', rlagent.hit - lfuagent.hit)
                            print('hitratio_RL=', rlagent.hit*1./totalreq)
                            print('hitratio_LRU=', lruagent.hit * 1. / totalreq)
                            print('hitratio_LFU=', lfuagent.hit * 1. / totalreq)
                            print('epochs_trained_RL=', rlagent.epoch)
                            print('tdloss:', rlagent.td_loss)
                            print('action:', rlagent.act_batch)
                            print('correlation:', sum(rlagent.corelation)/len(rlagent.corelation))
                            print('filename:', f)
                            print('='*80)

                            log_file.write(config_str + '\n')
                            log_file.write('CACHE_SIZE='
                                           + str(config.CACHE_SIZE) + '\t'
                                           + 'WINDOW_SIZES='
                                           + str(config.WINDOW_SIZES) + '\t'
                                           + 'PREFIX='
                                           + str(config.PREFIX) + '\n')
                            log_file.write('totalreq=' + str(totalreq) + '\n')
                            log_file.write('RL_hit-LRU_hit:' + str(rlagent.hit - lruagent.hit) + '\n')
                            log_file.write('RL_hit-LFU_hit:' + str(rlagent.hit - lfuagent.hit) + '\n')
                            log_file.write('hitratio_RL=' + str(rlagent.hit * 1. / totalreq) + '\n')
                            log_file.write('hitratio_LRU=' + str(lruagent.hit * 1. / totalreq) + '\n')
                            log_file.write('hitratio_LFU=' + str(lfuagent.hit * 1. / totalreq) + '\n')
                            log_file.write('epochs_trained_RL=' + str(rlagent.epoch) + '\n')
                            log_file.write('tdloss=' + str(rlagent.td_loss) + '\n')
                            log_file.write('action=' + str(rlagent.act_batch) + '\n')
                            log_file.write('correlation=' + str(sum(rlagent.corelation)/len(rlagent.corelation)) + '\n')
                            log_file.write('='*80 + '\n')
                            log_file.flush()

if __name__ == '__main__':
        main()