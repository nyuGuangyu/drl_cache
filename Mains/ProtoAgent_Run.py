from Agents import ProtoAgent
import pickle as cp
import numpy as np
import tensorflow as tf


CACHE_SIZE = 100
WINDOW_SIZES = [-100,-1000,-10000]
PREFIX = 18
Options = {'MF': False,
           'NoRandAct': False}

class StrToBytes:
    def __init__(self, fileobj):
        self.fileobj = fileobj
    def read(self, size):
        return self.fileobj.read(size).encode()
    def readline(self, size=-1):
        return self.fileobj.readline(size).encode()


def main():

    with tf.Session() as sess:

        central_agent = ProtoAgent.CentralAgent(sess, CACHE_SIZE, WINDOW_SIZES,
                                                norandact = Options['NoRandAct'])

        sess.run(tf.global_variables_initializer())

        totalreq_all = 0
        for day in range(8):

            LOG_FILE = './results/log_RL_test_'
            if Options['MF']:
                LOG_FILE += 'MF_'
            if Options['NoRandAct']:
                LOG_FILE += 'NoRandAct_'
            LOG_FILE += 'day' + str(day) + '_' + str(CACHE_SIZE) + '.csv'


            filename = './data/req/day' + str(day) + 'reqlist.cp'
            with open(filename,'rb') as infile:
                reqlist = cp.load(infile,encoding='bytes')
            if day != 0:
                if Options['MF']:
                    central_agent.NEAREST_K = 5
                else:
                    central_agent.NEAREST_K = 1
                filename = './data/neib/day' + str(day-1) + 'neighbors.cp'
                with open(filename) as infile:
                    neighbors = cp.load(infile)
            else:
                central_agent.NEAREST_K = 1
                filename = './data/neib/day' + str(0) + 'neighbors.cp'
                with open(filename,'rb') as infile:
                    neighbors = cp.load(infile,encoding='bytes')

            with open(LOG_FILE, 'wb') as log_file:
                for req in reqlist:
                    # cache update
                    central_agent.update_cache(req,neighbors)
                    totalreq_all += 1

                    if totalreq_all % 1000 == 0: # print result after xxx requests
                        print( 'CACHE_SIZE=',CACHE_SIZE,'WINDOW_SIZES=', WINDOW_SIZES,'PREFIX=',PREFIX)
                        print(Options)
                        print('totalreq=', totalreq_all)
                        print('totalhit=', np.sum(list(central_agent.hit.values())))
                        print( 'hitratio=', np.sum(list(central_agent.hit.values()))*1./totalreq_all)
                        print('hit_ratio_avg_sub=', np.mean(list(central_agent.hitratio.values())))
                        print( 'epochs_trained=', np.sum(list(central_agent.epoch.values())))
                        print('='*80)

                        log_file.write((str(totalreq_all) + '\t' +
                                       str(np.sum(list(central_agent.hit.values()))) + '\t' +
                                       str(np.sum(list(central_agent.hit.values())) * 1. / totalreq_all) + '\t' +
                                       str(np.mean(list(central_agent.hitratio.values()))) + '\t' +
                                       str(np.sum(list(central_agent.epoch.values()))) + '\n').encode())
                        log_file.flush()


if __name__ == '__main__':
        main()