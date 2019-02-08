"""
This code is for illustration purpose only.
Use multi_agent.py for better performance and speed.
"""

import numpy as np
import tensorflow as tf
from A3C import a3c
from scipy import stats

SUMMARY_DIR = './results'

class RLagent(object):
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



        self.S_INFO = self.CACHE_SIZE + 1  # from 0 to CACHE_SIZE
        self.S_LEN = 6  # shortLFU, midLFU, longLRU, shortLRU, midLRU, longLRU,

        self.rank = []

        self.A_DIM = self.CACHE_SIZE + 1

        self.TRAIN_SEQ_LEN = 1000
        self.QUEUE_LENGTH = -1*self.WINDOW_SIZES[-1]

        self.actor = a3c.ActorNetwork(sess,
                                      state_dim=[self.S_INFO, self.S_LEN],
                                      action_dim=self.A_DIM,
                                      learning_rate=actor_lr)

        self.critic = a3c.CriticNetwork(sess,
                                        state_dim=[self.S_INFO, self.S_LEN],
                                        learning_rate=critc_lr)

        self.summary_ops, self.summary_vars = a3c.build_summaries()
        self.writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)  # training monitor

        self.epoch = 0
        self.cache_state = np.zeros((self.S_INFO - 1, self.S_LEN))
        self.s_batch = []
        self.a_batch = []
        self.r_batch = []
        self.act_batch = []
        self.td_loss=0.
        self.actor_gradient_batch = []
        self.critic_gradient_batch = []
        self.cache_list = []
        self.replay_queue = []

        self.hit = 0
        self.totalreq = 0
        self.hitratio = 0.
        self.corelation = []

    def update_cache(self, req):
        if len(self.replay_queue) >= self.QUEUE_LENGTH:
            self.replay_queue.pop(0)
        self.replay_queue.append(req[2])
        self.totalreq += 1

        shortLFU, midLFU, longLFU = self.get_feat_LFU(req[2])
        shortLRU, midLRU, longLRU = self.get_feat_LRU(req[2])
        current_req_state = np.array(
                [shortLFU, midLFU, longLFU, shortLRU, midLRU, longLRU])
        state = np.vstack((self.cache_state, current_req_state))
        state = stats.zscore(state)

        reward = 0
        if req[2] in self.cache_list:
            reward = 10 #1
            self.hit += 1
            act = self.CACHE_SIZE # dont cache
            action = [0] * self.A_DIM
            action[act] = 1
        else:
            if len(self.cache_list) < self.CACHE_SIZE:  # if cache not full, direct cache.
                # update cache without using neighbor knowledge
                action = [0] * self.A_DIM
                act = len(self.cache_list)
                action[act] = 1

                self.cache_list.append(req[2])

                self.cache_state = np.vstack((np.array([list(self.get_feat_LFU(content)) + list(self.get_feat_LRU(content))
                                               for content in self.cache_list]),
                                              np.zeros((self.CACHE_SIZE - len(self.cache_list), self.S_LEN))))
            else:  # cache is full
                # actor's work
                # Receive observation state st
                action_prob = self.actor.predict(np.reshape(state, (1, state.shape[0], state.shape[1])))[0]
                # possible_act = action_prob.argsort()[-5:][::-1]#-5 top-k possible acts
                # possible_act = self.similar_acts(action_prob,state)
                # act = random.choice(list(possible_act))
                act = np.random.choice(range(101), p=action_prob)

                cor = np.corrcoef(action_prob, state[:,2])[0, 1]
                self.corelation.append(cor)

                # update cache
                if act != self.CACHE_SIZE:  # act = CACHE_SIZE means no update
                    cont_before = self.cache_list[act]
                    self.cache_list[act] = req[2]
                    cont_after = self.cache_list[act]
                    action = [0] * self.A_DIM
                    action[act] = 1
                else:
                    action = [0] * self.A_DIM
                    action[-1] = 1

                self.cache_state = np.vstack(
                    (np.array([list(self.get_feat_LFU(content)) + list(self.get_feat_LRU(content))
                               for content in self.cache_list]),
                     np.zeros((self.CACHE_SIZE - len(self.cache_list), self.S_LEN))))

        self.hitratio = self.hit*1./self.totalreq

        if len(self.r_batch) > self.TRAIN_SEQ_LEN:  ## do training once
            self.train()

        self.s_batch.append(state)
        self.a_batch.append(action)
        self.r_batch.append(reward)
        self.act_batch.append(act)

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

        summary_str = self.sess.run(self.summary_ops, feed_dict={

            self.summary_vars[0]: self.td_loss

        })

        self.writer.add_summary(summary_str, self.epoch)

        self.writer.flush()

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

    def similar_acts(self, action_prob, state, topK=6):
        max_prob_act = action_prob.argsort()[-1:][::-1]
        sims = []
        for i in range(state.shape[0]):
            compare_pos = state[i,:3]
            target_pos = state[max_prob_act,:3]
            dist = np.linalg.norm(compare_pos-target_pos)
            sims.append((i,dist))
        sims.sort(key=lambda x: x[1])

        return [k[0] for k in sims[:topK]]

    def get_feat_LRU(self,req_title):
        if req_title in list(reversed(self.replay_queue[self.WINDOW_SIZES[0]:])):
            s = list(reversed(self.replay_queue[self.WINDOW_SIZES[0]:])).index(req_title)
        else:
            s = -1 * self.WINDOW_SIZES[0]
        if req_title in list(reversed(self.replay_queue[self.WINDOW_SIZES[1]:])):
            m = list(reversed(self.replay_queue[self.WINDOW_SIZES[1]:])).index(req_title)
        else:
            m = -1 * self.WINDOW_SIZES[1]
        if req_title in list(reversed(self.replay_queue[self.WINDOW_SIZES[2]:])):
            l = list(reversed(self.replay_queue[self.WINDOW_SIZES[2]:])).index(req_title)
        else:
            l = -1 * self.WINDOW_SIZES[1]

        return s+1,m+1,l+1 # have to plus 1 to avoid return 0 that will later casue problem

    def get_feat_LFU(self,req_title):

        return np.array(
            [self.replay_queue[self.WINDOW_SIZES[0]:].count(req_title),  # short LFU
             self.replay_queue[self.WINDOW_SIZES[1]:].count(req_title),  # mid LFU
             self.replay_queue[self.WINDOW_SIZES[2]:].count(req_title)]) # long LFU




