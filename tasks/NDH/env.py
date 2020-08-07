''' Batched Room-to-Room navigation environment '''

import sys
sys.path.append('build')
import MatterSim
import csv
import numpy as np
import math
import base64
import json
import random
import networkx as nx
import utils

from utils import load_datasets, load_nav_graphs

csv.field_size_limit(sys.maxsize)



class EnvBatch():
    ''' A simple wrapper for a batch of MatterSim environments, 
        using discretized viewpoints and pretrained features '''

    def __init__(self, feature_store=None, batch_size=100, blind=False):
        if feature_store:
            print 'Loading image features from %s' % feature_store
            if blind:
                print("... and zeroing them out for 'blind' evaluation")
            tsv_fieldnames = ['scanId', 'viewpointId', 'image_w','image_h', 'vfov', 'features']
            self.features = {}
            with open(feature_store, "r+b") as tsv_in_file:
                reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = tsv_fieldnames)
                for item in reader:
                    self.image_h = int(item['image_h'])
                    self.image_w = int(item['image_w'])
                    self.vfov = int(item['vfov'])
                    long_id = self._make_id(item['scanId'], item['viewpointId'])
                    if not blind:
                        self.features[long_id] = np.frombuffer(base64.decodestring(item['features']),
                                dtype=np.float32).reshape((36, 2048))
                    else:
                        self.features[long_id] = np.zeros((36, 2048), dtype=np.float32)
        else:
            print 'Image features not provided'
            self.features = None
            self.image_w = 640
            self.image_h = 480
            self.vfov = 60
        self.batch_size = batch_size
        self.sims = []
        for i in range(batch_size):
            sim = MatterSim.Simulator()
            sim.setRenderingEnabled(False)
            sim.setDiscretizedViewingAngles(True)   # Set increment/decrement to 30 degree. (otherwise by radians)
            sim.setCameraResolution(self.image_w, self.image_h)
            sim.setCameraVFOV(math.radians(self.vfov))
            sim.initialize()
            self.sims.append(sim)

    def _make_id(self, scanId, viewpointId):
        return scanId + '_' + viewpointId   

    def newEpisodes(self, scanIds, viewpointIds, headings):
        for i, (scanId, viewpointId, heading) in enumerate(zip(scanIds, viewpointIds, headings)):
            # print("New episode %d" % i)
            # sys.stdout.flush()
            self.sims[i].newEpisode([scanId], [viewpointId], [heading], [0])
  
    def getStates(self):
        """
        Get list of states augmented with precomputed image features. rgb field will be empty.
        Agent's current view [0-35] (set only when viewing angles are discretized)
            [0-11] looking down, [12-23] looking at horizon, [24-35] looking up
        :return: [ ((30, 2048), sim_state) ] * batch_size
        """
        feature_states = []
        for i, sim in enumerate(self.sims):
            state = sim.getState()[0]
            long_id = self._make_id(state.scanId, state.location.viewpointId)
            if self.features:
                feature = self.features[long_id]     # Get feature for
                feature_states.append((feature, state))
            else:
                feature_states.append((None, state))
        return feature_states


    def makeActions(self, actions):
        ''' Take an action using the full state dependent action interface (with batched input).
            Every action element should be an (index, heading, elevation) tuple. '''
        for i, (index, heading, elevation) in enumerate(actions):
            self.sims[i].makeAction([index], [heading], [elevation])


class R2RBatch():
    ''' Implements the Room to Room navigation task, using discretized viewpoints and pretrained features '''

    def __init__(self, feature_store, batch_size=100, seed=10, splits=['train'], tokenizer=None,
                 path_type='planner_path', history='target', blind=False):
        self.buffered_state_dict = {}
        self.sim = utils.new_simulator()
        self.env = EnvBatch(feature_store=feature_store, batch_size=batch_size, blind=blind)
        self.data = []
        self.scans = []
        self.splits = splits
        for item in load_datasets(splits):
            double_num_dial = len(item['dialog_history'])
            dial = []
            dial_seps = []
            Last_QA = []
            QA_seps = []
            hist = []
            hist_enc = []
            target = []
            target.append(item['target'])
            tar_seps = []
            tar_seps.append('<TAR>')
            # For every dialog history, stitch together a single instruction string.
            self.scans.append(item['scan'])
            new_item = dict(item)
            new_item['inst_idx'] = item['inst_idx']
            if history == 'none':  # no language input at all
                new_item['instructions'] = ''
                if tokenizer:
                    new_item['instr_encoding'] = tokenizer.encode_sentence('')
            elif history == 'target' or len(item['dialog_history']) == 0:  # Have to use target only if no dialog history.
                tar = item['target']
                new_item['instructions'] = '<TAR> ' + tar
                new_item['Last_QA'] = Last_QA
                new_item['tar'] = item['target']
                if tokenizer:

                    new_item['instr_encoding'] = tokenizer.encode_sentence([tar], seps=['<TAR>'])
                    Last_QA_enc = tokenizer.encode_dial(None, None)
                    new_item['Last_QA_enc'] = Last_QA_enc

                    for i in range(15):
                        hist_enc.append(tokenizer.encode_dial(None, None))
                        hist.append('<pad>')
                    new_item['hist_enc'] = np.array(hist_enc)
                    new_item['hist'] = hist

                    tar_enc = tokenizer.encode_sentence(target, seps=tar_seps)
                    new_item['tar_enc'] = tar_enc
            elif history == 'oracle_ans':
                ora_a = item['dialog_history'][-1]['message']  # i.e., the last oracle utterance.
                tar = item['target']
                new_item['instructions'] = '<ORA> ' + ora_a + ' <TAR> ' + tar
                if tokenizer:
                    new_item['instr_encoding'] = tokenizer.encode_sentence([ora_a, tar], seps=['<ORA>', '<TAR>'])
            elif history == 'nav_q_oracle_ans':
                nav_q = item['dialog_history'][-2]['message']
                ora_a = item['dialog_history'][-1]['message']
                tar = item['target']
                new_item['instructions'] = '<NAV> ' + nav_q + ' <ORA> ' + ora_a + ' <TAR> ' + tar
                if tokenizer:
                    qa_enc = tokenizer.encode_sentence([nav_q, ora_a, tar], seps=['<NAV>', '<ORA>', '<TAR>'])
                    new_item['instr_encoding'] = qa_enc
            elif history == 'all':
                dia_inst = ''
                sentences = []
                seps = []
                for numm, turn in enumerate(item['dialog_history']):
                    sentences.append(turn['message'])
                    sep = '<NAV>' if turn['role'] == 'navigator' else '<ORA>'
                    seps.append(sep)
                    dia_inst += sep + ' ' + turn['message'] + ' '
                    dial.append(turn['message'])
                    dial_seps.append(sep)
                    if numm == double_num_dial - 1:
                        Last_QA.append(dial[-2])
                        Last_QA.append(dial[-1])
                        QA_seps.append(seps[-2])
                        QA_seps.append(seps[-1])
                sentences.append(item['target'])
                seps.append('<TAR>')
                dia_inst += '<TAR> ' + item['target']
                new_item['instructions'] = dia_inst
                new_item['Last_QA'] = Last_QA
                new_item['hist'] = hist
                new_item['tar'] = item['target']

                if tokenizer:
                    dia_enc = tokenizer.encode_sentence(sentences, seps=seps)
                    new_item['instr_encoding'] = dia_enc
                    ##
                    Last_QA_enc = tokenizer.encode_dial(Last_QA, seps=QA_seps)
                    new_item['Last_QA_enc'] = Last_QA_enc

                    tar_enc = tokenizer.encode_dial(target, seps=tar_seps)
                    new_item['tar_enc'] = tar_enc

                    dial_ix = 0
                    while dial_ix < (len(dial)-2):
                        qa_list = []
                        qa_seps = []

                        qa_list.append(dial[dial_ix])
                        qa_list.append(dial[dial_ix + 1])
                        qa_seps.append(dial_seps[dial_ix])
                        qa_seps.append(dial_seps[dial_ix + 1])

                        hist.append(qa_list)
                        hist_enc.append(tokenizer.encode_dial(qa_list, seps=qa_seps))
                        dial_ix = dial_ix + 2
                    if len(hist_enc) < 15:
                        for i in range(15-len(hist_enc)):
                            hist.append('<pad>')
                            hist_enc.append(tokenizer.encode_dial(None, None))
                        new_item['hist'] = hist
                        new_item['hist_enc'] = hist_enc

            # If evaluating against 'trusted_path', we need to calculate the trusted path and instantiate it.
            if path_type == 'trusted_path' and 'test' not in splits:
                # The trusted path is either the planner_path or the player_path depending on whether the player_path
                # contains the planner_path goal (e.g., stricter planner oracle success of player_path
                # indicates we can 'trust' it, otherwise we fall back to the planner path for supervision).
                # Hypothesize that this will combine the strengths of good human exploration with the known good, if
                # short, routes the planner uses.
                planner_goal = item['planner_path'][-1]  # this could be length 1 if "plan" is to not move at all.
                if planner_goal in item['player_path'][1:]:  # player walked through planner goal (did not start on it)
                    new_item['trusted_path'] = item['player_path'][:]  # trust the player.
                else:
                    new_item['trusted_path'] = item['planner_path'][:]  # trust the planner.
            self.data.append(new_item)

        self.scans = set(self.scans)
        self.splits = splits
        self.seed = seed
        random.seed(self.seed)
        random.shuffle(self.data)
        self.ix = 0
        self.batch_size = batch_size
        self._load_nav_graphs()
        self.path_type = path_type
        self.angle_feature = utils.get_all_point_angle_feature()
        print 'R2RBatch loaded with %d instructions, using splits: %s' % (len(self.data), ",".join(splits))

    def _load_nav_graphs(self):
        ''' Load connectivity graph for each scan, useful for reasoning about shortest paths '''
        print 'Loading navigation graphs for %d scans' % len(self.scans)
        self.graphs = load_nav_graphs(self.scans)
        self.paths = {}
        for scan,G in self.graphs.iteritems(): # compute all shortest paths
            self.paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        self.distances = {}
        for scan,G in self.graphs.iteritems(): # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def _next_minibatch(self):
        batch = self.data[self.ix:self.ix+self.batch_size]
        if len(batch) < self.batch_size:
            random.shuffle(self.data)
            self.ix = self.batch_size - len(batch)
            batch += self.data[:self.ix]
        else:
            self.ix += self.batch_size
        self.batch = batch

    def reset_epoch(self):
        ''' Reset the data index to beginning of epoch. Primarily for testing. 
            You must still call reset() for a new episode. '''
        self.ix = 0

    def _shortest_path_action(self, state, goalViewpointId):
        ''' Determine next action on the shortest path to goal, for supervised training. '''
        if state.location.viewpointId == goalViewpointId:
            return goalViewpointId      # Just stop here
        path = self.paths[state.scanId][state.location.viewpointId][goalViewpointId]
        nextViewpointId = path[1]
        return nextViewpointId

    def make_candidate(self, feature, scanId, viewpointId, viewId):
        def _loc_distance(loc):
            return np.sqrt(loc.rel_heading ** 2 + loc.rel_elevation ** 2)
        base_heading = (viewId % 12) * math.radians(30)
        adj_dict = {}
        long_id = "%s_%s" % (scanId, viewpointId)
        if long_id not in self.buffered_state_dict:
            for ix in range(36):
                if ix == 0:
                    self.sim.newEpisode([scanId], [viewpointId], [0], [math.radians(-30)])
                elif ix % 12 == 0:
                    self.sim.makeAction([0], [1.0], [1.0])
                else:
                    self.sim.makeAction([0], [1.0], [0])

                state = self.sim.getState()[0]
                assert state.viewIndex == ix

                # Heading and elevation for the viewpoint center
                heading = state.heading - base_heading
                elevation = state.elevation

                visual_feat = feature[ix]

                # get adjacent locations
                for j, loc in enumerate(state.navigableLocations[1:]):
                    # if a loc is visible from multiple view, use the closest
                    # view (in angular distance) as its representation
                    distance = _loc_distance(loc)

                    # Heading and elevation for for the loc
                    loc_heading = heading + loc.rel_heading
                    loc_elevation = elevation + loc.rel_elevation
                    angle_feat = utils.angle_feature(loc_heading, loc_elevation)
                    if (loc.viewpointId not in adj_dict or
                            distance < adj_dict[loc.viewpointId]['distance']):
                        adj_dict[loc.viewpointId] = {
                            'heading': loc_heading,
                            'elevation': loc_elevation,
                            "normalized_heading": state.heading + loc.rel_heading,
                            'scanId':scanId,
                            'viewpointId': loc.viewpointId, # Next viewpoint id
                            'pointId': ix,
                            'distance': distance,
                            'idx': j + 1,
                            'feature': np.concatenate((visual_feat, angle_feat), -1)
                        }
            candidate = list(adj_dict.values())
            self.buffered_state_dict[long_id] = [
                {key: c[key]
                 for key in
                    ['normalized_heading', 'elevation', 'scanId', 'viewpointId',
                     'pointId', 'idx']}
                for c in candidate
            ]
            return candidate
        else:
            candidate = self.buffered_state_dict[long_id]
            candidate_new = []
            for c in candidate:
                c_new = c.copy()
                ix = c_new['pointId']
                normalized_heading = c_new['normalized_heading']
                visual_feat = feature[ix]
                loc_heading = normalized_heading - base_heading
                c_new['heading'] = loc_heading
                angle_feat = utils.angle_feature(c_new['heading'], c_new['elevation'])
                c_new['feature'] = np.concatenate((visual_feat, angle_feat), -1)
                c_new.pop('normalized_heading')
                candidate_new.append(c_new)
            return candidate_new

    def _get_obs(self):
        obs = []
        for i,(feature,state) in enumerate(self.env.getStates()):
            item = self.batch[i]
            base_view_id = state.viewIndex
            candidate = self.make_candidate(feature, state.scanId, state.location.viewpointId, state.viewIndex)
            feature = np.concatenate((feature, self.angle_feature[base_view_id]), -1)
            obs.append({
                'inst_idx': item['inst_idx'],
                'scan': state.scanId,
                'viewpoint': state.location.viewpointId,
                'viewIndex': state.viewIndex,
                'heading': state.heading,
                'elevation': state.elevation,
                'feature': feature,
                "candidate": candidate,
                'step': state.step,
                'navigableLocations': state.navigableLocations,
                'instructions': item['instructions'],
                'teacher': self._shortest_path_action(state, item[self.path_type][-1]) if 'test' not in self.splits else None,
            })
            if 'instr_encoding' in item:
                obs[-1]['instr_encoding'] = item['instr_encoding']
                obs[-1]['Last_QA_enc'] = item['Last_QA_enc']
                obs[-1]['hist_enc'] = item['hist_enc']
                obs[-1]['tar_enc'] = item['tar_enc']
        return obs

    def reset(self):
        ''' Load a new minibatch / episodes. '''
        self._next_minibatch()
        scanIds = [item['scan'] for item in self.batch]
        if 'test' not in self.splits:
            viewpointIds = [item[self.path_type][0] for item in self.batch]
        else:
            viewpointIds = [item['start_pano']['pano'] for item in self.batch]

        headings = [item['start_pano']['heading'] for item in self.batch]
        self.env.newEpisodes(scanIds, viewpointIds, headings)
        return self._get_obs()   

    def step(self, actions):
        ''' Take action (same interface as makeActions) '''
        self.env.makeActions(actions)
        return self._get_obs()


