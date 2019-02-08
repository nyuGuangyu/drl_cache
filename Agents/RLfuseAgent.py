"""
This code is for illustration purpose only.
Use multi_agent.py for better performance and speed.
"""

import numpy as np
import tensorflow as tf
from A3C import a3c
from scipy import stats
from Agents import LRUAgent, LFUAgent
from easydict import EasyDict


SUMMARY_DIR = './results'

class RLfuseAgent(object):
    """
    Input: Subgroup_ID, Current Request, Cache Size = Int, Nearest K,\
                Similarity Matrix = dict(), Window Sizes = [short,mid,long] ...
    Output: Hit, TotalReq, Hitratio ...
    """

    def __init__(self, sess, config):
        self.sess = sess
        self.config = config
        self.PREFIX = config.PREFIX
        self.CACHE_SIZE = config.CACHE_SIZE
        self.WINDOW_SIZES = config.WINDOW_SIZES
        actor_lr = config.actor_lr
        critc_lr = config.critc_lr

        self.shadow_lru = LRUAgent.LRUAgent(config) # shadow lru agent
        self.shadow_lfu = LFUAgent.LFUAgent(config) # shadow lru agent

        self.A_DIM = config.act_num
        self.cand_weights = np.linspace(.0, 1.0, num=self.A_DIM) # 11 candidate weights between lru and lfu
        self.S_INFO = config.feat_num # lru & lfu performances
        self.S_LEN = config.prev_slot_num # previous y slots performances
        self.TRAIN_SEQ_LEN = config.TRAIN_SEQ_LEN
        self.collect_freq = config.collect_freq # every x req collect performances once, thus every x requests get a new performance and act once

        self.actor = a3c.ActorNetwork(sess,
                                      state_dim=[self.S_INFO, self.S_LEN],
                                      action_dim=self.A_DIM,
                                      learning_rate=actor_lr)

        self.critic = a3c.CriticNetwork(sess,
                                        state_dim=[self.S_INFO, self.S_LEN],
                                        learning_rate=critc_lr)

        self.epoch = 0
        self.cache_state = np.zeros((self.S_INFO, self.S_LEN))
        self.s_batch = []
        self.a_batch = []
        self.r_batch = []
        self.act_batch = []
        self.r_per_batch = 0

        self.td_loss = 0.
        self.actor_gradient_batch = []
        self.critic_gradient_batch = []
        self.cache_list = []

        # init lru & lfu propotion to 0.5 & 0.5
        lru_quota = int(np.floor(.5 * self.CACHE_SIZE))
        config_lru = EasyDict(config.copy())
        config_lru.CACHE_SIZE = lru_quota
        self.lru_part = LRUAgent.LRUAgent(config_lru)
        self.lru_part.cache_list = self.shadow_lru.cache_list[-1 * lru_quota:]  # for lru, get most recent m content

        lfu_quota = int(np.floor(.5 * self.CACHE_SIZE))
        config_lfu = EasyDict(config.copy())
        config_lfu.CACHE_SIZE = lfu_quota
        cnts = [(content, self.shadow_lfu.replay_queue.count(content)) for content in
                self.shadow_lfu.cache_list]
        self.lfu_part = LFUAgent.LFUAgent(config_lfu)
        self.lfu_part.cache_list = [cnt[0] for cnt in sorted(cnts, key=lambda x: x[1], reverse=True)[
                                   :lfu_quota]]  # for lfu, take the most freqent n content

        self.replay_queue = []

        self.hit = 0
        self.totalreq = 0
        self.hitratio = 0.

        self.act_flag = False

    def update_cache(self,req):

        # before performances complete, no action.
        if not len(self.shadow_lru.performances) == self.config.prev_slot_num or not len(self.shadow_lfu.performances) == self.config.prev_slot_num:
            self.shadow_lru.update_cache(req)
            self.shadow_lfu.update_cache(req)
            return

        if self.totalreq % self.config.collect_freq == 0: # every x act once

            self.act_flag = True

            state = np.vstack((self.shadow_lru.performances, self.shadow_lfu.performances))
            # state = stats.zscore(state,axis=1)

            action_prob = self.actor.predict(np.reshape(state, (1, state.shape[0], state.shape[1])))[0] # shape=[2,S_LEN]
            act_pos = np.random.choice(range(self.A_DIM), p=action_prob)
            act = self.cand_weights[act_pos] # act in [0.0,1.0]
            action = [0] * self.A_DIM
            action[act_pos] = 1

            # reset a new lru & lfu part
            lru_quota = int(np.floor(act * self.CACHE_SIZE))
            config_lru = EasyDict(self.config.copy())
            config_lru.CACHE_SIZE = lru_quota
            self.lru_part = LRUAgent.LRUAgent(config_lru)
            self.lru_part.cache_list = self.shadow_lru.cache_list[-1 * lru_quota:] # for lru, get most recent m content

            lfu_quota = int(np.ceil((1 - act) * self.CACHE_SIZE))
            config_lfu = EasyDict(self.config.copy())
            config_lfu.CACHE_SIZE = lfu_quota
            cnts = [(content, self.shadow_lfu.replay_queue.count(content)) for content in self.shadow_lfu.cache_list + [req[2]]]
            self.lfu_part = LFUAgent.LFUAgent(config_lfu)
            self.lfu_part.cache_list = [cnt[0] for cnt in sorted(cnts, key=lambda x: x[1], reverse=True)[
                                   :lfu_quota]] # for lfu, take the most freqent n content

            self.cache_list = self.lru_part.cache_list + self.lfu_part.cache_list

            if len(self.r_batch) > self.TRAIN_SEQ_LEN:  ## do training once after xx actions
                self.train()

            self.s_batch.append(state)
            self.a_batch.append(action)
            self.r_batch.append(self.r_per_batch)
            self.act_batch.append(act)

            self.r_per_batch = 0

        self.totalreq += 1

        # now update lru & lfu, but no need to take propotion actions.
        # since cache list comes from lru and lfu, it start as full, no need to check if it's full.
        if req[2] in self.cache_list:
        # if req[2] in self.shadow_lfu.cache_list or req[2] in self.shadow_lru.cache_list: # test performance upperbound
            self.r_per_batch += 1  # 1
            self.hit += 1

        # assign all job to two sub policies.
        self.lru_part.update_cache(req)
        self.lfu_part.update_cache(req)

        self.shadow_lru.update_cache(req)
        self.shadow_lfu.update_cache(req)


    def train(self):
        self.epoch += 1
        actor_gradient, critic_gradient, td_batch = \
            a3c.compute_gradients(s_batch=np.stack(self.s_batch[1:], axis=0),  # ignore the first chuck
                                  a_batch=np.vstack(self.a_batch[1:]),  # since we don't have the
                                  r_batch=np.vstack(self.r_batch[1:]),  # control over it
                                  actor=self.actor, terminal=False, critic=self.critic)
        self.td_loss = np.mean(td_batch)

        self.actor_gradient_batch.append(actor_gradient)
        self.critic_gradient_batch.append(critic_gradient)

        if len(self.actor_gradient_batch) >= 2:

            assert len(self.actor_gradient_batch) == len(self.critic_gradient_batch)
            for i in range(len(self.actor_gradient_batch)):
                self.actor.apply_gradients(self.actor_gradient_batch[i])
                self.critic.apply_gradients(self.critic_gradient_batch[i])

            self.actor_gradient_batch = []
            self.critic_gradient_batch = []

        del self.s_batch[:]
        del self.a_batch[:]
        del self.r_batch[:]
        del self.act_batch[:]


