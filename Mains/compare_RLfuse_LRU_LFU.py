import sys; sys.path.append('/home/lgy/drl_cache')
from Agents import LRUAgent, LFUAgent, RLfuseAgent
import pickle as cp
import tensorflow as tf
from utils.util import get_config_from_json, choose_group, index_contents
from os import listdir
import random

try:
    config_file = "../configs/config_RLfuse.json"
    config, _ = get_config_from_json(config_file)
except Exception as e:
    print("missing or invalid arguments %s" % e)
    exit(0)

def main():

    data_dir = config.reqs_dir_real_indexed
    if not listdir(data_dir):
        index_contents(data_dir)
    target_group = choose_group(data_dir)

    with tf.Session() as sess:

        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        lruagent = LRUAgent.LRUAgent(config)
        lfuagent = LFUAgent.LFUAgent(config)
        rlfagent = RLfuseAgent.RLfuseAgent(sess, config)

        sess.run(tf.global_variables_initializer())
        totalreq = 0

        with open(config.LOG_FILE, 'wb') as log_file:
            for f in sorted(listdir(data_dir)): # use only 1 file
                with open(data_dir + f, 'rb') as infile:
                    reqlist = cp.load(infile)

                # for req in reqlist:
                while(True): # train forever
                    # sample sub lists
                    # seed = random.randint(0, len(reqlist)-rlagent.TRAIN_SEQ_LEN-10)
                    # req_samples = reqlist[seed : seed + rlagent.TRAIN_SEQ_LEN]

                    for req in reqlist:
                        # focus on only one target group
                        if req[1][:config.PREFIX] != target_group:
                            continue
                        rlfagent.update_cache(req)
                        lruagent.update_cache(req)
                        lfuagent.update_cache(req)
                        # rl fuse not start yet, reset lru and lfu hit&cnt to 0
                        if not rlfagent.act_flag:
                            lruagent.hit = 0
                            lfuagent.hit = 0
                            lruagent.totalreq = 0
                            lfuagent.totalreq = 0
                        totalreq += 1

                        if totalreq % 1000 == 0: # print result every xxx requests
                            if not rlfagent.totalreq:
                                continue
                            print('CACHE_SIZE=', config.CACHE_SIZE,'WINDOW_SIZES=', config.WINDOW_SIZES,'PREFIX=', config.PREFIX)
                            print('totalreq=', totalreq)
                            print('hitratio_RL=', rlfagent.hit * 1. / rlfagent.totalreq)
                            print('hitratio_LRU=', lruagent.hit * 1. / lruagent.totalreq)
                            print('hitratio_LFU=', lfuagent.hit * 1. / lfuagent.totalreq)
                            print('epochs_trained_RL=', rlfagent.epoch)
                            print('tdloss:', rlfagent.td_loss)
                            print('action:', ["%.1f" % v for v in rlfagent.act_batch])
                            print('filename:', f)
                            print('='*80)

                            log_file.write('CACHE_SIZE='
                                           + str(config.CACHE_SIZE) + '\t'
                                           + 'WINDOW_SIZES='
                                           + str(config.WINDOW_SIZES) + '\t'
                                           + 'PREFIX='
                                           + str(config.PREFIX) + '\n')
                            log_file.write('totalreq=' + str(totalreq) + '\n')
                            log_file.write('RL_hit-LRU_hit:' + str(rlfagent.hit - lruagent.hit) + '\n')
                            log_file.write('RL_hit-LFU_hit:' + str(rlfagent.hit - lfuagent.hit) + '\n')
                            if rlfagent.totalreq and lfuagent.totalreq and lfuagent.totalreq:
                                log_file.write('hitratio_RL=' + str(rlfagent.hit * 1. / rlfagent.totalreq) + '\n')
                                log_file.write('hitratio_LRU=' + str(lruagent.hit * 1. / lruagent.totalreq) + '\n')
                                log_file.write('hitratio_LFU=' + str(lfuagent.hit * 1. / lfuagent.totalreq) + '\n')
                            else:
                                log_file.write('hitratio unavailable' + '\n')
                            log_file.write('epochs_trained_RL=' + str(rlfagent.epoch) + '\n')
                            log_file.write('tdloss=' + str(rlfagent.td_loss) + '\n')
                            log_file.write('action=' + str(rlfagent.act_batch) + '\n')
                            log_file.write('='*80 + '\n')
                            log_file.flush()

if __name__ == '__main__':
        main()