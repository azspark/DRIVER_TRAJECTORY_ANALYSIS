import os
import sys
import math
import numpy as np
import pandas as pd
import geopy
from collections import Counter

from .trajectory import Trajectory

class DrivingState:
    """the drving state transition graph for one trajectory
    """
    def __init__(self, coords, timestamps, driver_id):
        """Calculate and store the feature of single trajectory

        Args:
            coords (np.ndarray,): [(lon1, lat1), (lon2, lat2), ...]
            timestamps (np.ndarray): unix time, same length as coords
            driver_id (np.ndarray): driver id for this trajectory
        """
        self.distances = None
        self.speeds = None
        self.angles = None
        self.accelerations = None
        self.delta_theta = None

        self.coords = np.array(coords)
        self.timestamps = np.array(timestamps)
        self.driver_id = np.array(driver_id)

        self.get_angles()
        self.get_delta_theta()
        self.get_distance()
        self.get_speed() #pay attention to it, this will delete the abnormal point
        self.get_accleration()

        self.transition_graph = None
        self.transition_sequence = None
        self.driving_state = {'acceleration':0, 'constant':1, 'deceleration':2}
        self.direction_state = {'straight': 3, 'turn': 4}
        self.state2id = dict()
        idx = 0
        for state in self.driving_state.values():
            for direction in self.direction_state.values():
                self.state2id[(state, direction)] = idx
                idx += 1
    
    def get_seq_vector(self):
        """transpose the seq to the vector

        Returns:
            np.array: dim 9
        """
        self.transition_sequence = self.get_seq()
        id_seq = map(lambda x : self.state2id[x], self.transition_sequence)
        count = Counter(id_seq)
        v = np.zeros(len(self.state2id))
        # import pdb; pdb.set_trace()
        for key in count:
            v[key] = count[key]
        if np.sum(v) != 0.0:
            norm_v = (v - np.min(v)) / (np.max(v) - 0)
        else:
            norm_v = v
        return norm_v
    
    def get_graph_vector(self):
        """transpose the edge of the graph to vector

        Returns:
            np.array: dim 36
        """
        self.transition_graph = self.get_graph()
        graph_vec = self.transition_graph.reshape(-1)
        if np.sum(graph_vec) != 0.0:
            normed_graph_vec = (graph_vec - np.min(graph_vec)) / (np.max(graph_vec) - 0)
        else:
            normed_graph_vec = graph_vec
        
        return normed_graph_vec


    def get_seq(self):
        """generate seq with element (drivingstateid, directionid)

        Returns:
            list: [(1,4), (2,4), (3,6)...]
        """
        if self.transition_sequence is not None:
            return self.transition_sequence
        self.transition_sequence = []
        s = self.driving_state['constant']
        a = self.direction_state['straight']
        for state, dir in zip(self.accelerations, self.delta_theta):
            if state > 0.1:
                s = self.driving_state['acceleration']
            elif state < -0.1:
                s = self.driving_state['deceleration']
            else:
                s = self.driving_state['constant']

            if dir > math.pi / 6:
                a = self.direction_state['turn']
            else:
                a = self.direction_state['straight']
            self.transition_sequence.append((s,a))
        return self.transition_sequence

    def get_graph(self):
        """generate the transition graph from the driving state sequence

        Returns:
            np.array: (state_dim, state_dim)
        """
        if self.transition_graph is not None:
            return self.transition_graph
        self.transition_sequence = self.get_seq()
        self.transition_graph = np.zeros((len(self.state2id), len(self.state2id)))
        for idx, state in enumerate(self.transition_sequence[:-1]):
            state_id_src = self.state2id[self.transition_sequence[idx]]
            state_id_tar = self.state2id[self.transition_sequence[idx+1]]
            self.transition_graph[state_id_src][state_id_tar] += 1
        # self state transition is set to 0.
        # for i in range(len(self.transition_graph)):
        #     self.transition_graph[i][i] = 0
        return self.transition_graph


        

    def get_distance(self):
        if self.distances is not None:
            return self.distances
        prev_coords = self.coords[:-1]
        after_coords = self.coords[1:]
        # geopy.distance.distace accept (lat, lon) input
        # import pdb; pdb.set_trace()
        distances = [geopy.distance.distance((prev[1], prev[0]), (after[1], after[0])).meters for prev, after in zip(prev_coords, after_coords)]
        self.distances = np.array(distances)
        return self.distances

    def get_speed(self):
        if self.speeds is not None:
            return self.speeds
        self.time_diff = self.timestamps[1:] - self.timestamps[:-1]
        distances = self.get_distance()
        speeds = distances / self.time_diff
        speeds = np.concatenate((np.array([0.0]), speeds), axis=0)  # add zero speed for the first point
        normal_idx = speeds < 38.9  # 140km/h, 140 / 3.6 = 38.8888
        if False in normal_idx:  # directly delete the abnormal part
            # print(normal_idx)
            self.angles = self.get_angles()
            self.angles = self.angles[normal_idx]
            self.delta_theta = self.delta_theta[normal_idx]
            speeds = speeds[normal_idx]
            self.distances = distances[normal_idx[1:]]
            self.time_diff = self.time_diff[normal_idx[1:]]
            self.coords = self.coords[normal_idx]
        self.speeds = speeds

        return self.speeds

    def get_accleration(self):
        if self.accelerations is not None:
            return self.accelerations
        speeds = self.get_speed()
        speeds_diff = speeds[1:] - speeds[:-1]
        self.accelerations = np.concatenate((np.array([0.0]), (speeds_diff / self.time_diff)), axis=0)  # add zero accleration for the first point
        # print('accelerations')
        # print(np.max(self.accelerations))
        # print(np.mean(self.accelerations))
        return self.accelerations

    def get_angles(self):
        if self.angles is not None:
            return self.angles
        self.distances = self.get_distance()
        angles = []
        for idx, coord in enumerate(self.coords[:-1]):
            # angles.append(np.arccos(np.dot(coord,self.coords[idx+1])/(np.linalg.norm(coord)*np.linalg.norm(self.coords[idx+1]))))  # no, arccos is not good for this
            # import pdb; pdb.set_trace()
            if self.distances[idx] >= 0.1:  # it's meaningless to compute the angle when drving real small distance
                coord_diff = self.coords[idx+1] - coord
                if coord_diff[0] == 0:
                    coord_diff[0] = 1e-8
                theta = np.arctan(coord_diff[1] / coord_diff[0])
                if coord_diff[1] < 0:
                    theta = math.pi + theta
                angles.append(theta)
            else:
                if len(angles) > 0:
                    angles.append(angles[-1])
                else:
                    angles.append(0.0)
        try:
            self.angles = angles + [angles[-1]]  # add for last point
        except Exception as e:
            self.angles = angles + [0.0]
        self.angles = np.array(self.angles)
        return self.angles

    def get_delta_theta(self):
        """get the delta direction for each timestamp

        """
        if self.delta_theta is not None:
            return self.delta_theta
        self.angles = self.get_angles()
        delta = []
        for idx, angle in enumerate(self.angles[:-1]):
            tmp_angle = abs(self.angles[idx+1] - angle)
            tmp2 = math.pi*2 - self.angles[idx+1] - angle
            if abs(tmp2) < abs(tmp_angle):
                tmp_angle = abs(tmp2)
            delta.append(tmp_angle)
        self.delta_theta = np.array([0] + delta)

        # print(f'direction: {np.max(self.delta_theta)}')
        # print(f'direction: {np.mean(self.delta_theta)}')
        
        return self.delta_theta


