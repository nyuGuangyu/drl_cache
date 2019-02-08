"""
if combine lru and lfu, we want to know the optimal eviction lies on which ranks on both lru and lfu.
Plot then in two dimension figure, x-axis: lru ranks, y-axis: lfu ranks.
"""

import cPickle as cp
import numpy as np
from Agents import FiFAgent, LRUAgent, LFUAgent
from utils.util import get_config_from_json, choose_group
from collections import Counter
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


outfile = 'plot_fif_ranks.cp'

try:
    samples = cp.load(open(outfile,'rb'))
except:
    # read in the data: day0 indexed
    data_file = '../data/req_index/day0reqlist.cp'
    reqlist = cp.load(open(data_file, 'rb'))

    # configs
    config_file = "../configs/config_RLfuse.json"
    config, _ = get_config_from_json(config_file)

    target_group = choose_group('../data/req_index/')

    reqlist = [e for e in reqlist if e[1][:config.PREFIX] == target_group]

    fifagent = FiFAgent.FiFAgent(config, reqlist)
    lruagent = LRUAgent.LRUAgent(config)
    lfuagent = LFUAgent.LFUAgent(config)
    cnt_lfu = Counter()

    samples = []
    totalreq = 0

    for i, req in enumerate(reqlist):

        fifagent.update_cache(req,i)
        lruagent.update_cache(req)
        lfuagent.update_cache(req)
        cnt_lfu.update([req[2]]) # counter only takes list
        totalreq += 1

        if not fifagent.cur_evict:
            continue

        # if cur_evict is not none, extract two ranks from lru and lfu using the replay queues
        opt_ev = fifagent.cur_evict
        lru_rnk = list(reversed(lruagent.replay_queue)).index(opt_ev) # index func gives the first match
        sorted_cnt_lfu = sorted(cnt_lfu.items(), key=lambda x:x[1], reverse=True) # rank larger means less freqent
        lfu_rnk = sorted_cnt_lfu.index((opt_ev, cnt_lfu[opt_ev])) # find the index of (opt_ev, its counts)

        samples.append((lru_rnk,lfu_rnk))

        if totalreq % 1000 == 0:  # print result every xxx requests
            print('CACHE_SIZE=', config.CACHE_SIZE, 'WINDOW_SIZES=', config.WINDOW_SIZES, 'PREFIX=', config.PREFIX)
            print('totalreq=', totalreq)
            print('samples=', len(samples))
            print('hitratio_FiF=', fifagent.hit * 1. / fifagent.totalreq)
            print('hitratio_LRU=', lruagent.hit * 1. / lruagent.totalreq)
            print('hitratio_LFU=', lfuagent.hit * 1. / lfuagent.totalreq)
            print('=' * 80)

    cp.dump(samples,open(outfile,'wb'))

# plot
colors = (0,0,0)
area = np.pi*0.3
x = [sample[0] for sample in samples]
y = [sample[1] for sample in samples]
fig, ax = plt.subplots()
ax = plt.scatter(x[:1000], y[:1000], s=area, c=colors, alpha=0.5)
plt.gca().invert_yaxis()
plt.gca().xaxis.tick_top()
plt.xlim(-50, 150)
plt.ylim(150, -50)
plt.title('fif in lru&lfu ranks', y=1.08)
plt.xlabel('lru rank')
plt.ylabel('lfu rank')
plt.show()
plt.savefig("plot_fif_ranks.pdf")






