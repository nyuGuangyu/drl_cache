"""
This code is for illustration purpose only.
Use multi_agent.py for better performance and speed.
"""

import numpy as np
import tensorflow as tf
from A3C import a3c
import random


SUMMARY_DIR = './results'

class CentralAgent(object):
    """
    Input: Subgroup_ID, Current Request, Cache Size = Int, Nearest K,\ 
                Similarity Matrix = dict(), Window Sizes = [short,mid,long] ...
    Output: Hit, TotalReq, Hitratio ...
    """

    def __init__(self, sess, cache_size,window_sizes,norandact=True,actor_lr=0.001, critc_lr=0.001):
        self.norandact = norandact
        self.sess = sess
        self.PREFIX = 18
        self.CACHE_SIZE = cache_size
        self.NEAREST_K = None
        self.WINDOW_SIZES = window_sizes

        self.S_INFO = self.CACHE_SIZE + 1  # from 0 to CACHE_SIZE
        self.S_LEN = 3  # short-, mid-, long-,

        self.rank = []

        self.A_DIM = self.CACHE_SIZE + 1

        self.TRAIN_SEQ_LEN = 100
        self.QUEUE_LENGTH = -1*self.WINDOW_SIZES[-1]

        self.actor = a3c.ActorNetwork(sess,
                                      state_dim=[self.S_INFO, self.S_LEN], action_dim=self.A_DIM,
                                      learning_rate=actor_lr)

        self.critic = a3c.CriticNetwork(sess,
                                        state_dim=[self.S_INFO, self.S_LEN],
                                        learning_rate=critc_lr)

        self.summary_ops, self.summary_vars = a3c.build_summaries()
        self.writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)  # training monitor

        self.epoch = {}
        self.cache_state = {}
        self.s_batch = {}
        self.a_batch = {}
        self.r_batch = {}
        self.act_batch = {}
        self.entropy_record = {}
        self.td_loss=0.

        self.switch_flag = False

        self.actor_gradient_batch = {}
        self.critic_gradient_batch = {}

        self.cache_list = {}
        self.replay_queue = {}

        self.hit = {}
        self.totalreq = {}
        self.hitratio = {}

    def update_cache(self, req, neighbors):

        subid = req[1][:self.PREFIX]

        if subid not in self.replay_queue:
            self.state_init(subid,req)
        else:
            if len(self.replay_queue[subid]) > self.QUEUE_LENGTH:
                self.replay_queue[subid].pop(0)
            self.replay_queue[subid].append(req[2])
        self.totalreq[subid] += 1

        if subid in neighbors:
            sim_sub_ids = neighbors[subid][:self.NEAREST_K]
        else:
            sim_sub_ids = [(subid,1.)] + [(subid,0.)] * (self.NEAREST_K - 1)

        current_req_state = self.cf_state(req[2],sim_sub_ids)
        state = np.vstack((self.cache_state[subid], current_req_state))


        reward = 0
        if req[2] in self.cache_list[subid]:
            reward = 10 #1
            self.hit[subid] += 1
            act = self.CACHE_SIZE # dont cache
            action = [0] * self.A_DIM
            action[act] = 1
        else:
            if len(self.cache_list[subid]) < self.CACHE_SIZE:  # if cache not full, direct cache.
                # update cache without using neighbor knowledge
                action = [0] * self.A_DIM
                act = len(self.cache_list[subid])
                action[act] = 1

                self.cache_list[subid].append(req[2])

                self.cache_state[subid] = np.vstack(([self.cf_state(content,sim_sub_ids)
                                          for content in self.cache_list[subid]],
                                         np.zeros((self.CACHE_SIZE - len(self.cache_list[subid]), 3))))
            else:  # cache is full
                # actor's work
                # Receive observation state st
                action_prob = self.actor.predict(np.reshape(state, (1,
                                                                    state.shape[0],
                                                                    state.shape[1])))
                # possible_act = action_prob[0].argsort()[-5:][::-1]#-5 top-k possible acts
                possible_act = self.similar_acts(action_prob[0],state)
                act = random.choice(list(possible_act))
                # act = np.random.choice(range(101), p=action_prob[0])
                if act != 100:
                    self.rank.append(self.cache_list[subid][act])
                else:
                    self.rank.append(req[2])

                self.entropy_record[subid].append(a3c.compute_entropy(action_prob[0]))
                # update cache
                if act != self.CACHE_SIZE:  # act = CACHE_SIZE means no update
                    cont_before = self.cache_list[subid][act]
                    self.cache_list[subid][act] = req[2]
                    cont_after = self.cache_list[subid][act]
                    action = [0] * self.A_DIM
                    action[act] = 1
                else:
                    action = [0] * self.A_DIM
                    action[-1] = 1

                self.cache_state[subid] = [self.cf_state(content,sim_sub_ids)
                                          for content in self.cache_list[subid]]

        self.hitratio[subid] = self.hit[subid]*1./self.totalreq[subid]

        if len(self.r_batch[subid]) > self.TRAIN_SEQ_LEN:  ## do training once
            self.train(subid)

        self.s_batch[subid].append(state)
        self.a_batch[subid].append(action)
        self.r_batch[subid].append(reward)
        self.act_batch[subid].append(act)

    def train(self,subid):
        self.epoch[subid] += 1
        actor_gradient, critic_gradient, td_batch = \
            a3c.compute_gradients(s_batch=np.stack(self.s_batch[subid][1:], axis=0),  # ignore the first chuck
                                  a_batch=np.vstack(self.a_batch[subid][1:]),  # since we don't have the
                                  r_batch=np.vstack(self.r_batch[subid][1:]),  # control over it
                                  actor=self.actor, terminal=False, critic=self.critic)
        self.td_loss = np.mean(td_batch)

        self.actor_gradient_batch[subid].append(actor_gradient)
        self.critic_gradient_batch[subid].append(critic_gradient)

        summary_str = self.sess.run(self.summary_ops, feed_dict={

            self.summary_vars[0]: self.td_loss

        })

        self.writer.add_summary(summary_str, self.epoch[subid])

        self.writer.flush()

        self.entropy_record[subid] = []

        if len(self.actor_gradient_batch[subid]) >= 16:

            assert len(self.actor_gradient_batch[subid]) == len(self.critic_gradient_batch[subid])
            for i in range(len(self.actor_gradient_batch[subid])):
                self.actor.apply_gradients(self.actor_gradient_batch[subid][i])
                self.critic.apply_gradients(self.critic_gradient_batch[subid][i])

            self.actor_gradient_batch[subid] = []
            self.critic_gradient_batch[subid] = []

        del self.s_batch[subid][:]
        del self.a_batch[subid][:]
        del self.r_batch[subid][:]
        del self.act_batch[subid][:]

    def cf_state(self,content,sim_sub_ids):
        state_list = []
        for (subid,sim) in sim_sub_ids:
            s = np.array([self.replay_queue[subid][self.WINDOW_SIZES[0]:].count(content),
                 self.replay_queue[subid][self.WINDOW_SIZES[1]:].count(content),
                 self.replay_queue[subid][self.WINDOW_SIZES[2]:].count(content)])
            state_list.append(s*sim)

        return sum(state_list)

    def state_init(self,subid,req):
        self.epoch[subid] = 0
        self.cache_state[subid] = np.zeros((self.S_INFO - 1, self.S_LEN))
        current_req_state = [1] * self.S_LEN
        state = np.vstack((self.cache_state[subid], current_req_state))  # combine cache state and
        # request state
        # to get state
        self.s_batch[subid] = [state]
        self.a_batch[subid] = [[1] + [0] * (self.A_DIM-1)]
        self.r_batch[subid] = [0]
        self.act_batch[subid] = [0]
        self.entropy_record[subid] = []

        self.actor_gradient_batch[subid] = []
        self.critic_gradient_batch[subid] = []

        self.cache_list[subid] = [req[2]]
        self.replay_queue[subid] = [req[2]]

        self.hit[subid] = 0
        self.totalreq[subid] = 1

    def similar_acts(self, action_prob, state):
        max_prob_act = action_prob.argsort()[-1:][::-1]
        sims = []
        for i in range(state.shape[0]):
            compare_pos = state[i,:]
            target_pos = state[max_prob_act,:]
            dist = np.linalg.norm(compare_pos-target_pos)
            sims.append((i,dist))
        sims.sort(key=lambda x: x[1])

        return [k[0] for k in sims[:6]]

