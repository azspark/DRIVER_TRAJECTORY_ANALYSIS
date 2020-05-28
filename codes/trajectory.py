import numpy as np
import pandas as pd
from geopy.distance import distance
import folium
from .utils import base_stat
from collections import Counter
import math


class Trajectory:
    """Calculate and store the feature of single trajectory"""

    extracted_feature = ['']
    def __init__(self, coords, timestamps, driver_id, traj_id, turning_threshold=0.3, 
        use_graph_feature=False, acc_threshold=0.1, base_feat=True):
        """
        Args:
        - coords: np.ndarray, [(lon1, lat1), (lon2, lat2), ...]
        - timestamps: np.ndarray, unix time, same length as coords
        - driver_id: np.ndarray,
        - base_feat: bool, if calculate basic feature such as speed, accleration
        """
        self.distances = None
        self.speeds = None
        self.angles = None
        self.accelerations = None
        self.with_noise_point = False
        self.too_much_noise = False

        self.turning_threshold = turning_threshold
        self.acc_threshold = acc_threshold
        self.use_graph_feature = use_graph_feature
        self.coords = np.array(coords)
        self.timestamps = timestamps
        self.driver_id = driver_id
        self.traj_id = traj_id
        if base_feat:
            self.get_accleration()

        if base_feat and not self.too_much_noise:  # won't use too noisy traj
            self.get_angles()
            self._driving_state()
            if self.use_graph_feature:
                self._init_graph_feat_param()


    def get_distance(self):
        if self.distances is not None:
            return self.distances
        prev_coords = self.coords[:-1]
        after_coords = self.coords[1:]
        # geopy.distance.distace accept (lat, lon) input
        distances = [distance((prev[1], prev[0]), (after[1], after[0])).meters for prev, after in zip(prev_coords, after_coords)]
        self.distances = np.array(distances)
        return self.distances

    def get_speed(self):
        if self.speeds is not None:
            return self.speeds
        self.time_diff = self.timestamps[1:] - self.timestamps[:-1]
        distances = self.get_distance()
        speeds = distances / self.time_diff
        speeds = np.concatenate((np.array([0.0]), speeds), axis=0)  # add zero speed for the first point
        normal_idx = speeds < 38.9  # 140km/h, 140 / 3.6 = 38.8888; dtype=boolean65
        normal_coord_num = sum(normal_idx)
        if normal_coord_num < 5 or sum(normal_idx) / float(len(normal_idx)) < 0.5:
            self.too_much_noise = True
        if False in normal_idx:  # directly delete the abnormal part
            # print(normal_idx)
            self.with_noise_point = True
            speeds = speeds[normal_idx]
            self.distances = distances[normal_idx[1:]]
            self.time_diff = self.time_diff[normal_idx[1:]]
            self.coords = self.coords[normal_idx]
            self.timestamps = self.timestamps[normal_idx]
        self.speeds = speeds
        return self.speeds

    def get_accleration(self):
        if self.accelerations is not None:
            return self.accelerations
        speeds = self.get_speed()
        speeds_diff = speeds[1:] - speeds[:-1]
        self.accelerations = np.concatenate((np.array([0.0]), (speeds_diff / self.time_diff)), axis=0)  # add zero accleration for the first point
        return self.accelerations

    # def get_angles(self):
    #     angles = []
    #     for idx, coord in enumerate(self.coords[:-1]):
    #         # angles.append(np.arccos(np.dot(coord,self.coords[idx+1])/(np.linalg.norm(coord)*np.linalg.norm(self.coords[idx+1]))))  # no, arccos is not good for this
            
    #         if self.distances[idx] >= 0.1:  # it's meaningless to compute the angle when drving real small distance
    #             coord_diff = self.coords[idx+1] - coord
    #             theta = np.arctan(coord_diff[1] / coord_diff[0])
    #             if coord_diff[1] < 0:
    #                 theta = np.math.pi + theta
    #             angles.append(theta)
    #         else:
    #             if len(angles) > 0:
    #                 angles.append(angles[-1])
    #             else:
    #                 angles.append(0.0)
    #     self.angles = np.array(angles + [angles[-1]])  # add for last point
    #     return self.angles

    def get_angles(self):
        if self.angles is not None:
            return self.angles
        angles = []
        for idx, coord in enumerate(self.coords[:-1]):
            # angles.append(np.arccos(np.dot(coord,self.coords[idx+1])/(np.linalg.norm(coord)*np.linalg.norm(self.coords[idx+1]))))  # no, arccos is not good for this
            # import pdb; pdb.set_trace()
            if self.distances[idx] >= 0.1:  # it's meaningless to compute the angle when drving real small distance
                coord_diff = self.coords[idx+1] - coord
                if coord_diff[0] == 0:
                    coord_diff[0] = 1e-8
                theta = np.arctan(coord_diff[1] / coord_diff[0])
                # if coord_diff[1] < 0:
                #     theta = math.pi + theta
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

    def get_basic_feature(self):

        out = [self.too_much_noise]
        if not self.too_much_noise:
            out += base_stat(self.speeds)
            out += base_stat(self.accelerations, count_pos_neg=True)
            if sum(self.is_turning) > 0:
                out += base_stat(self.speeds[self.is_turning])
                out += base_stat(self.accelerations[self.is_turning], count_pos_neg=True)
            else:
                out += [0.0 for i in range(15)]
        else:
            out += [0.0 for i in range(30)]
        return out
    
    def get_graph_feature(self):
        if not self.too_much_noise:
            return list(self.get_seq_vector()) + list(self.get_graph_vector())
        else:
            return [self.too_much_noise] + [0.0 for i in range(36)]

    def _driving_state(self):
        """judge car is turning or not"""
        angle_diff = self.angles[1:] - self.angles[:-1]
        self.angle_diff = np.concatenate((np.array([0.0]), angle_diff), axis=0)

        self.is_turning = np.absolute(self.angle_diff) > self.turning_threshold


    def vis(self, with_marker=False, osm_map=None, time=None, mark_od_point=True, color='blue'):
        """Visualization of this single trajectory
        
        Used in jupyter notebook.
        Attention: folium accept trajectories in (lat, lon) format.
        Args:
        - with_marker: boolean, if true, driving details of each point will be shown on traj with marker
        - mark_od_point: if true will display origin and destion with marker no matter 'with_marker'
        """
        start_point = self.coords[0]
        traj = [(cd[1], cd[0]) for cd in self.coords]
        if osm_map is None:
            m = folium.Map(location=traj[0], zoom_start=13)
        else:
            m = osm_map
        line_tooltip="Driver:%d, index:%d" % (self.driver_id, self.traj_id)
        if time is not None:
            line_tooltip += ',' + str(time)
        folium.PolyLine(locations=traj, color=color, tooltip=line_tooltip).add_to(m)
        if mark_od_point:
            start_tooltip = 'Start' + str(self.driver_id) + ':' + str(self.traj_id)
            end_tooltip = 'End' + str(self.driver_id) + ':' + str(self.traj_id)
            folium.Marker(location=traj[0], tooltip=start_tooltip, icon=folium.map.Icon(color='orange')).add_to(m)
            folium.Marker(location=traj[-1], tooltip=end_tooltip, icon=folium.map.Icon(color='lightblue')).add_to(m)
        if with_marker:
            len_traj = len(traj)
            for idx, cd in enumerate(traj):
                if mark_od_point and (idx == 0 or idx == len_traj-1):
                    continue
                tool_tip = "Driver:%d;pid:%d<br />speed:%.2fm/s acceleration:%.4f angles:%.4f" % (self.driver_id, idx, 
                    self.speeds[idx], self.accelerations[idx], self.angles[idx])
                tool_tip += "turning:%s" % str(self.is_turning[idx])
                folium.Marker(location=cd, tooltip=tool_tip, icon=folium.map.Icon(color=color)).add_to(m)
        return m


    def _init_graph_feat_param(self):
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
        for state, dir in zip(self.accelerations, self.angle_diff):
            if state > self.acc_threshold:
                s = self.driving_state['acceleration']
            elif state < -self.acc_threshold:
                s = self.driving_state['deceleration']
            else:
                s = self.driving_state['constant']

            if np.absolute(dir) > self.turning_threshold:
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

    def get_speed_info(self):
        """Return speed and steering speed along with time.(API for CityCells)"""
        return self.speeds, self.timestamps, self.speeds[self.is_turning], self.timestamps[self.is_turning]

        
    # def get_delta_theta(self):
    #     """get the delta direction for each timestamp

    #     """
    #     if self.delta_theta is not None:
    #         return self.delta_theta
    #     self.angles = self.get_angles()
    #     delta = []
    #     for idx, angle in enumerate(self.angles[:-1]):
    #         tmp_angle = abs(self.angles[idx+1] - angle)
    #         tmp2 = math.pi*2 - self.angles[idx+1] - angle
    #         if abs(tmp2) < abs(tmp_angle):
    #             tmp_angle = tmp2
    #         delta.append(tmp_angle)
    #     self.delta_theta = np.array([0] + delta)

    #     # print(f'direction: {np.max(self.delta_theta)}')
    #     # print(f'direction: {np.mean(self.delta_theta)}')
        
    #     return self.delta_theta