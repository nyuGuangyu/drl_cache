"""
This code is for illustration purpose only.
Use multi_agent.py for better performance and speed.
"""

import numpy as np
import random

class LFUAgent(object):
    """
    Input: Subgroup_ID, Current Request, Cache Size = Int, Nearest K,\
                Similarity Matrix = dict(), Window Sizes = [short,mid,long] ...
    Output: Hit, TotalReq, Hitratio ...
    """

    def __init__(self, config):
        self.config = config
        self.PREFIX = config.PREFIX
        self.CACHE_SIZE = config.CACHE_SIZE
        self.replay_queue_len = config.WINDOW_SIZES[-1] * -1 #

        self.cache_list = []

        self.hit = 0
        self.totalreq = 0
        self.hitratio = 0.
        self.replay_queue = []
        self.performances = []
        self.slot_hit = 0

    def update_cache(self, req):
        self.totalreq += 1
        if len(self.replay_queue) >= self.replay_queue_len:
            self.replay_queue.pop(0)
        self.replay_queue.append(req[2])

        if self.totalreq % self.config.collect_freq == 0:
            self.performances.append(self.slot_hit)
            if len(self.performances) > self.config.prev_slot_num:
                self.performances.pop(0)
            self.slot_hit = 0

        if req[2] in self.cache_list: # put recent accessed on top of the cachelist
            self.hit += 1
            self.slot_hit += 1
        else:
            if len(self.cache_list) < self.CACHE_SIZE:  # if cache not full, direct cache.
                self.cache_list.append(req[2])
            else:  # cache is full and req not hit
                cnts = [(i,self.replay_queue.count(content)) for i,content in enumerate(self.cache_list + [req[2]])]
                evict_pos = sorted(cnts, key=lambda x:x[1])[0][0]
                if evict_pos != len(self.cache_list):
                    self.cache_list[evict_pos] = req[2]

        self.hitratio = self.hit*1./self.totalreq



