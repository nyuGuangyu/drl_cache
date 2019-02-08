from Agents import LRUAgent, ProtoAgent
import pickle as cp
import numpy as np
import tensorflow as tf

CACHE_SIZE = 100
WINDOW_SIZES = [-100, -1000, -10000]
Options = {'MF': False,
           'NoRandAct': False,
           'actor_lr': 0.01,
           'critc_lr': 0.1
           }
PREFIX = 18


def main():

    with tf.Session() as sess:

        central_agent = ProtoAgent.CentralAgent(sess, CACHE_SIZE, WINDOW_SIZES,
                                                norandact = Options['NoRandAct'],
                                                actor_lr=Options['actor_lr'],
                                                critc_lr=Options['critc_lr'])

        lru_agent = LRUAgent.LRUAgent(CACHE_SIZE)

        sess.run(tf.global_variables_initializer())


        totalreq_all = 0

        for day in range(8):

            local_request = 0
            LOG_FILE = './results/log_compare_CentralAg_LRU_test_'
            if Options['MF']:
                LOG_FILE += 'MF_'
            if Options['NoRandAct']:
                LOG_FILE += 'NoRandAct_'
            LOG_FILE += 'day' + str(day) + '_' + str(CACHE_SIZE) + '.csv'

            if day != 0:
                if Options['MF']:
                    central_agent.NEAREST_K = 5
                else:
                    central_agent.NEAREST_K = 1
                filename = './data/neib/day' + str(day-1) + 'neighbors.cp'
                with open(filename,'rb') as infile:
                    neighbors = cp.load(infile, encoding='bytes')
            else:
                central_agent.NEAREST_K = 1
                filename = './data/neib/day' + str(0) + 'neighbors.cp'
                with open(filename,'rb') as infile:
                    neighbors = cp.load(infile)

            #filename = './data/req/day' + str(day) + 'reqlist.cp'
            filename = './data/fakedReq/day' + str(day) + 'reqlist.cp'
            with open(filename,'rb') as infile:
                reqlist = cp.load(infile)

            with open(LOG_FILE, 'wb') as log_file:
                # reqs = group_requests(reqlist)
                for req in reqlist:
                    # cache update
                    totalreq_all += 1
                    if totalreq_all >20000:
                        central_agent.switch_flag = False
                    else:
                        central_agent.switch_flag = False

                    central_agent.update_cache(req,neighbors)
                    lru_agent.update_cache(req)

                    local_request += 1
                    if local_request == 95000:
                        log_file.flush()
                        break


                    if totalreq_all % 1000 == 0: # print result every xxx requests
                        print('CACHE_SIZE=',CACHE_SIZE,'WINDOW_SIZES=', WINDOW_SIZES,'PREFIX=',PREFIX)
                        print(Options)
                        print('totalreq=', totalreq_all)
                        # print 'totalhit_RL=', np.sum(central_agent.hit.values())
                        # print 'totalhit_LRU=', np.sum(lru_agent.hit.values())
                        print( 'RL_hit - LRU_hit:',np.sum(list(central_agent.hit.values())) - np.sum(list(lru_agent.hit.values())))
                        print('hitratio_RL=', np.sum(list(central_agent.hit.values()))*1./totalreq_all)
                        print( 'hitratio_LRU=', np.sum(list(lru_agent.hit.values())) * 1. / totalreq_all)
                        print('hit_ratio_avg_sub_RL=', np.mean(list(central_agent.hitratio.values())))
                        print( 'hit_ratio_avg_sub_LRU=', np.mean(list(lru_agent.hitratio.values())))
                        print('epochs_trained_RL=', np.sum(list(central_agent.epoch.values())))
                        print('tdloss:', central_agent.td_loss)
                        print('action:', central_agent.act_batch)
                        print('action:', central_agent.rank[-30:])
                        print('day:',str(day))
                        print('='*80)

                        log_file.write((str(totalreq_all) + '\t' +
                                        str(np.sum(list(central_agent.hit.values())) - np.sum(list(lru_agent.hit.values()))) + '\t' +
                                       str(np.sum(list(central_agent.hit.values()))) + '\t' +
                                       str(np.sum(list(central_agent.hit.values())) * 1. / totalreq_all) + '\t' +
                                       str(np.mean(list(central_agent.hitratio.values()))) + '\t' +
                                       str(np.sum(list(central_agent.epoch.values()))) + '\t' +
                                       str(np.sum(list(lru_agent.hit.values()))) + '\t' +
                                       str(np.sum(list(lru_agent.hit.values())) * 1. / totalreq_all) + '\t' +
                                       str(np.mean(list(lru_agent.hitratio.values()))) + '\n').encode())
                        log_file.flush()

if __name__ == '__main__':
        main()