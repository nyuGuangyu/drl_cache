"""
This code is for illustration purpose only.
Use multi_agent.py for better performance and speed.
"""

import numpy as np
import random

class FiFAgent(object):
    """
    Input: Subgroup_ID, Current Request, Cache Size = Int, Nearest K,\
                Similarity Matrix = dict(), Window Sizes = [short,mid,long] ...
    Output: Hit, TotalReq, Hitratio ...
    """

    def __init__(self, config, reqlist):
        # fif needs to know the future reqlist and current req index, also assume the end is the end of day0.
        self.config = config
        self.PREFIX = config.PREFIX
        self.CACHE_SIZE = config.CACHE_SIZE

        self.cache_list = []

        self.reqlist = [req[2] for req in reqlist]

        self.hit = 0
        self.totalreq = 0
        self.hitratio = 0.
        self.performances = []
        self.slot_hit = 0
        self.cur_evict = None


    def update_cache(self, req, cur):
        self.totalreq += 1

        if self.totalreq % self.config.collect_freq == 0:
            self.performances.append(self.slot_hit)
            if len(self.performances) > self.config.prev_slot_num:
                # only keep track of recent x slots performances
                self.performances.pop(0)
            self.slot_hit = 0

        if req[2] in self.cache_list: # if hit, do nothing
            self.hit += 1
            self.slot_hit += 1
            self.cur_evict = None
        else:
            if len(self.cache_list) < self.CACHE_SIZE:  # if cache not full, direct cache.
                self.cache_list.append(req[2])
            else:  # cache is full and req not hit, evict the farthest in future
                future = self.reqlist[cur+1:]
                clist = []
                for c in self.cache_list:
                    if c not in future:
                        self.cur_evict = c
                        self.cache_list.remove(c)
                        self.cache_list.append(req[2])
                        return
                    else:
                        clist.append((c,future.index(c))) # .index find the first match in list
                self.cur_evict = sorted(clist,key=lambda x:x[1])[-1][0]
                self.cache_list.remove(self.cur_evict)
                self.cache_list.append(req[2])

        self.hitratio = self.hit*1./self.totalreq



