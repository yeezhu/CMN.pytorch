''' Agents: stop/random/shortest/seq2seq  '''

import json
import os
import sys
import numpy as np
import random
import time
import math

import torch
import torch.nn as nn
import torch.distributions as D
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import utils
from env import R2RBatch
from utils import padding_idx
from param import args

class BaseAgent(object):
    ''' Base class for an R2R agent to generate and save trajectories. '''

    def __init__(self, env, results_path):
        self.env = env
        self.results_path = results_path
        random.seed(1)
        self.results = {} 
        self.losses = [] # For learning agents
    
    def write_results(self):
        output = [{'inst_idx': k, 'trajectory': v} for k, v in self.results.iteritems()]
        with open(self.results_path, 'w') as f:
            json.dump(output, f)

    def rollout(self):
        ''' Return a list of dicts containing instr_id:'xx', path:[(viewpointId, heading_rad, elevation_rad)]  '''
        raise NotImplementedError

    @staticmethod
    def get_agent(name):
        return globals()[name+"Agent"]

    def test(self):
        self.env.reset_epoch()
        self.losses = []
        self.results = {}
        # We rely on env showing the entire batch before repeating anything
        #print 'Testing %s' % self.__class__.__name__
        looped = False
        while True:
            for traj in self.rollout():
                if traj['inst_idx'] in self.results:
                    looped = True
                else:
                    self.results[traj['inst_idx']] = traj['path']
            if looped:
                break

    
class StopAgent(BaseAgent):  
    ''' An agent that doesn't move! '''

    def rollout(self):
        traj = [{
            'inst_idx': ob['inst_idx'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in self.env.reset()]
        return traj


class RandomAgent(BaseAgent):
    ''' An agent that picks a random direction then tries to go straight for
        five viewpoint steps and then stops. '''

    def rollout(self):
        obs = self.env.reset()
        traj = [{
            'inst_idx': ob['inst_idx'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in obs]
        self.steps = random.sample(range(-11,1), len(obs))
        ended = [False] * len(obs)
        # for t in range(30):  # 20 ep len + 10 (as in MP); planner paths
        for t in range(130):  # 120 ep len + 10 (emulating above); player paths
            actions = []
            for i,ob in enumerate(obs):
                if self.steps[i] >= 5:
                    actions.append((0, 0, 0)) # do nothing, i.e. end
                    ended[i] = True
                elif self.steps[i] < 0:
                    actions.append((0, 1, 0)) # turn right (direction choosing)
                    self.steps[i] += 1
                elif len(ob['navigableLocations']) > 1:
                    actions.append((1, 0, 0)) # go forward
                    self.steps[i] += 1
                else: 
                    actions.append((0, 1, 0)) # turn right until we can go forward
            obs = self.env.step(actions)
            for i,ob in enumerate(obs):
                if not ended[i]:
                    traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))
        return traj


class ShortestAgent(BaseAgent):
    ''' An agent that always takes the shortest path to goal. '''

    def rollout(self):
        obs = self.env.reset()
        traj = [{
            'inst_idx': ob['inst_idx'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in obs]
        ended = np.array([False] * len(obs))
        while True:
            actions = [ob['teacher'] for ob in obs]
            obs = self.env.step(actions)
            for i,a in enumerate(actions):
                if a == (0, 0, 0):
                    ended[i] = True
            for i,ob in enumerate(obs):
                if not ended[i]:
                    traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))
            if ended.all():
                break
        return traj


class Seq2SeqAgent(BaseAgent):
    ''' An agent based on an LSTM seq2seq model with attention. '''

    # For now, the agent can't pick which forward move to make - just the one in the middle
    model_actions = ['left', 'right', 'up', 'down', 'forward', '<end>', '<start>', '<ignore>']
    env_actions = {
      "left": ([0],[-1], [0]), # left
      "right": ([0], [1], [0]), # right
      "up": ([0], [0], [1]), # up
      "down": ([0], [0],[-1]), # down
      "forward": ([1], [0], [0]), # forward
      "<end>": ([0], [0], [0]), # <end>
      "<start>": ([0], [0], [0]), # <start>
      "<ignore>": ([0], [0], [0])  # <ignore>
    }
    feedback_options = ['teacher', 'argmax', 'sample']

    def __init__(self, env, results_path, encoder, decoder, episode_len=20, path_type='planner_path'):
        super(Seq2SeqAgent, self).__init__(env, results_path)
        self.feature_size = 2048
        self.encoder = encoder
        self.decoder = decoder
        self.episode_len = episode_len
        self.losses = []
        # self.criterion = nn.CrossEntropyLoss(ignore_index = self.model_actions.index('<ignore>'))
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignoreid, size_average=False)

    @staticmethod
    def n_inputs():
        return len(Seq2SeqAgent.model_actions)

    @staticmethod
    def n_outputs():
        return len(Seq2SeqAgent.model_actions)-2 # Model doesn't output start or ignore

    def _sort_batch(self, obs):
        ''' Extract instructions from a list of observations and sort by descending
            sequence length (to enable PyTorch packing). '''

        seq_tensor = np.array([ob['instr_encoding'] for ob in obs])
        seq_lengths = np.argmax(seq_tensor == padding_idx, axis=1)
        seq_lengths[seq_lengths == 0] = seq_tensor.shape[1] # Full length

        seq_tensor = torch.from_numpy(seq_tensor)
        seq_lengths = torch.from_numpy(seq_lengths)

        # Sort sequences by lengths
        seq_lengths, perm_idx = seq_lengths.sort(0, True)
        sorted_tensor = seq_tensor[perm_idx]
        mask = (sorted_tensor == padding_idx)[:,:seq_lengths[0]]

        return Variable(sorted_tensor, requires_grad=False).long().cuda(), \
               mask.byte().cuda(), \
               list(seq_lengths), list(perm_idx)

    def _candidate_variable(self, obs):
        candidate_leng = [len(ob['candidate']) + 1 for ob in obs]       # +1 is for the end
        candidate_feat = np.zeros((len(obs), max(candidate_leng), self.feature_size + args.angle_feat_size), dtype=np.float32)
        # Note: The candidate_feat at len(ob['candidate']) is the feature for the END
        # which is zero in my implementation
        for i, ob in enumerate(obs):
            for j, c in enumerate(ob['candidate']):
                candidate_feat[i, j, :] = c['feature']                         # Image feat
        return torch.from_numpy(candidate_feat).cuda(), candidate_leng

    def get_input_feat(self, obs):
        input_a_t = np.zeros((len(obs), args.angle_feat_size), np.float32)
        for i, ob in enumerate(obs):
            input_a_t[i] = utils.angle_feature(ob['heading'], ob['elevation'])
        input_a_t = torch.from_numpy(input_a_t).cuda()

        f_t = self._feature_variable(obs)      # Image features from obs
        candidate_feat, candidate_leng = self._candidate_variable(obs)

        return input_a_t, f_t, candidate_feat, candidate_leng

    def _feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        features = np.empty((len(obs), args.num_view, self.feature_size + args.angle_feat_size), dtype=np.float32)
        # feature_size = obs[0]['feature'].shape[0]
        # features = np.empty((len(obs), self.feature_size), dtype=np.float32)
        for i,ob in enumerate(obs):
            features[i,:] = ob['feature']
        return Variable(torch.from_numpy(features), requires_grad=False).cuda()

    def _teacher_action(self, obs, ended):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:                                            # Just ignore this index
                a[i] = args.ignoreid
            else:
                for k, candidate in enumerate(ob['candidate']):
                    if candidate['viewpointId'] == ob['teacher']:   # Next view point
                        a[i] = k
                        break
                else:   # Stop here
                    try:
                        assert ob['teacher'] == ob['viewpoint']         # The teacher action should be "STAY HERE"
                    except:
                        import pdb; pdb.set_trace()
                    a[i] = len(ob['candidate'])
        return torch.from_numpy(a).cuda()

    def make_equiv_action(self, a_t, perm_obs, perm_idx=None, traj=None):
        """
        Interface between Panoramic view and Egocentric view
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """
        def take_action(i, idx, name):
            if type(name) is int:       # Go to the next view
                self.env.env.sims[idx].makeAction([name], [0], [0])
            else:                       # Adjust
                self.env.env.sims[idx].makeAction(*self.env_actions[name])
            state = self.env.env.sims[idx].getState()[0]
            if traj is not None:
                traj[i]['path'].append((state.location.viewpointId, state.heading, state.elevation))
        if perm_idx is None:
            perm_idx = range(len(perm_obs))
        for i, idx in enumerate(perm_idx):
            action = a_t[i]
            if action != -1:            # -1 is the <stop> action
                select_candidate = perm_obs[i]['candidate'][action]
                src_point = perm_obs[i]['viewIndex']
                trg_point = select_candidate['pointId']
                src_level = (src_point ) // 12   # The point idx started from 0
                trg_level = (trg_point ) // 12
                while src_level < trg_level:    # Tune up
                    take_action(i, idx, 'up')
                    src_level += 1
                while src_level > trg_level:    # Tune down
                    take_action(i, idx, 'down')
                    src_level -= 1
                while self.env.env.sims[idx].getState()[0].viewIndex != trg_point:    # Turn right until the target
                    take_action(i, idx, 'right')
                assert select_candidate['viewpointId'] == \
                       self.env.env.sims[idx].getState()[0].navigableLocations[select_candidate['idx']].viewpointId
                take_action(i, idx, select_candidate['idx'])

    def rollout(self):
        obs = np.array(self.env.reset())
        batch_size = len(obs)

        # Reorder the language input for the encoder
        seq, seq_mask, seq_lengths, perm_idx = self._sort_batch(obs)
        perm_obs = obs[perm_idx]

        # Record starting point
        traj = [{
            'inst_idx': ob['inst_idx'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in perm_obs]

        # Forward through encoder, giving initial hidden state and memory cell for decoder
        Last_QA_tensor = np.array([obb['Last_QA_enc'] for obb in perm_obs])
        Last_QA_lengths = np.argmax(Last_QA_tensor == padding_idx, axis=1)
        Last_QA_lengths[Last_QA_lengths == 0] = Last_QA_tensor.shape[1]  # Full length
        Last_QA_lengths = torch.from_numpy(Last_QA_lengths)
        Last_QA_lengths = Last_QA_lengths.long().cuda()
        Last_QA_tensor = torch.from_numpy(Last_QA_tensor)
        Last_QA = Variable(Last_QA_tensor, requires_grad=False).long().cuda()

        H = []
        H_l = []
        for i, obbb in enumerate(perm_obs):
            H.append([])
            H_l.append([])
            for j in range(15):
                H[i].append(obbb['hist_enc'][j])
                h = np.array([obbb['hist_enc'][j]])
                h_l = np.argmax(h == padding_idx, axis=1)
                if h_l == 0:
                    H_l[i].append(1)
                else:
                    H_l[i].append(h_l)
        hist_tensor = np.array(H)
        hist_tensor = torch.from_numpy(hist_tensor)
        hist = Variable(hist_tensor, requires_grad=False).long().cuda()
        hist_lengths = np.array(H_l)
        hist_lengths = torch.from_numpy(hist_lengths)
        hist_lengths = hist_lengths.long().cuda()

        tar_tensor = np.array([obbbb['tar_enc'] for obbbb in perm_obs])
        tar_lengths = np.ones(batch_size)
        tar_lengths = torch.from_numpy(tar_lengths)
        tar_lengths = tar_lengths.long().cuda()
        tar_tensor = torch.from_numpy(tar_tensor)
        tar = Variable(tar_tensor, requires_grad=False).long().cuda()

        ctx,h_t,c_t = self.encoder(seq, seq_lengths, Last_QA, Last_QA_lengths, hist, hist_lengths, tar, tar_lengths) # Initial action
        
        # Last_QA_mask = utils.length2mask(Last_QA_lengths.cpu())
        a_t = Variable(torch.ones(batch_size).long() * self.model_actions.index('<start>'), 
                    requires_grad=False).cuda()
        ended = np.array([False] * batch_size) # Indices match permuation of the model, not env

        visited = [set() for _ in perm_obs]

        # Do a sequence rollout and calculate the loss
        self.loss = 0
        env_action = [None] * batch_size
        h1 = h_t
        for t in range(self.episode_len):
            input_a_t, f_t, candidate_feat, candidate_leng = self.get_input_feat(perm_obs)
            candidate_mask = utils.length2mask(candidate_leng)
            f_t = self._feature_variable(perm_obs) # Image features from obs
            h_t, c_t, logit, h1 = self.decoder(input_a_t, f_t, candidate_feat, 
                                    h_t, h1, c_t, ctx, None)

            if 'test' in self.env.splits:
                for ob_id, ob in enumerate(perm_obs):
                    visited[ob_id].add(ob['viewpoint'])
                    for c_id, c in enumerate(ob['candidate']):
                        if c['viewpointId'] in visited[ob_id]:
                            candidate_mask[ob_id][c_id] = 1

            logit.masked_fill_(candidate_mask, -float('inf'))

            # Supervised training
            if 'test' not in self.env.splits:
                target = self._teacher_action(perm_obs, ended)
                # self.loss += self.criterion(logit, target)
                tmp_loss = self.criterion(logit, target)
                if not math.isinf(tmp_loss):
                    self.loss += tmp_loss

            # Determine next model inputs
            if self.feedback == 'teacher': 
                a_t = target                # teacher forcing
            elif self.feedback == 'argmax': 
                _,a_t = logit.max(1)        # student forcing - argmax
                a_t = a_t.detach()
            elif self.feedback == 'sample':
                probs = F.softmax(logit, dim=1)
                m = D.Categorical(probs)
                a_t = m.sample()            # sampling an action from model
            else:
                sys.exit('Invalid feedback option')

            cpu_a_t = a_t.cpu().numpy()
            for i, next_id in enumerate(cpu_a_t):
                if next_id == (candidate_leng[i]-1) or next_id == args.ignoreid:    # The last action is <end>
                    cpu_a_t[i] = -1   

            self.make_equiv_action(cpu_a_t, perm_obs, perm_idx, traj)
            obs = np.array(self.env._get_obs())
            perm_obs = obs[perm_idx]


            ended[:] = np.logical_or(ended, (cpu_a_t == -1))
            # Early exit if all ended
            if ended.all(): 
                break
        if 'test' not in self.env.splits:
            self.losses.append(self.loss.item() / self.episode_len)
        return traj

    def test(self, use_dropout=False, feedback='argmax', allow_cheat=False):
        ''' Evaluate once on each instruction in the current environment '''
        if not allow_cheat: # permitted for purpose of calculating validation loss only
            assert feedback in ['argmax', 'sample'] # no cheating by using teacher at test time!
        self.feedback = feedback
        if use_dropout:
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()
        super(Seq2SeqAgent, self).test()

    def train(self, encoder_optimizer, decoder_optimizer, n_iters, feedback='teacher'):
        ''' Train for a given number of iterations '''
        assert feedback in self.feedback_options
        self.feedback = feedback
        self.encoder.train()
        self.decoder.train()
        self.losses = []
        for iter in range(1, n_iters + 1):
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            self.rollout()
            self.loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

    def save(self, encoder_path, decoder_path):
        ''' Snapshot models '''
        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.decoder.state_dict(), decoder_path)

    def load(self, encoder_path, decoder_path):
        ''' Loads parameters (but not training state) '''
        self.encoder.load_state_dict(torch.load(encoder_path))
        self.decoder.load_state_dict(torch.load(decoder_path))

