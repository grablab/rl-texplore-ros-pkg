import glob
import os, pickle
import random

import pandas as pd
import numpy as np

from itertools import product


class ContactState:
    def __init__(self, left, right, G, classifier='random'):
        self.left = left  # discretized contact 0-9
        self.right = right
        self._neighbors = dict()
        self._in = dict()
        self._out = dict()
        self.G = G
        if classifier == 'random':
            self.classifier = lambda s, t: random.choice([True, False])
        else:
            self.classifier = classifier

    def construct_neighbors(self, n=1, classifier=None):
        if self._neighbors:
            return
        classifier = classifier or self.classifier
        if not classifier:
            classifier = lambda s, t: random.choice([True, False])
        neighbors = self.visited_neighbors.copy()
        skipped = {(self.left, self.right)}
        for step in range(1, n + 1):
            left_step_up, left_step_down = min(self.left + step, 9), max(self.left - step, 0)
            right_step_up, right_step_down = min(self.right + step, 9), max(self.right - step, 0)
            for l, r in product([left_step_up, left_step_down, self.left],
                                [right_step_up, right_step_down, self.right]):
                # check if classifier allows path to this node from current state
                if not classifier(self.coords, (l, r)):
                    skipped.add((l, r))
                    continue
                # check if we have already visited or skipped this node
                # checking skipped again in case of random classifier
                if (l, r) in neighbors or (l, r) in skipped:
                    continue
                cs = self.G.get((l, r), ContactState(l, r, self.G))
                if (l, r) not in self.G:
                    self.G[(l, r)] = cs
                neighbors[cs] = 0
        self._neighbors = neighbors

    @property
    def neighbors(self):
        if not self._neighbors:
            self.construct_neighbors()
        return self._neighbors

    @property
    def visited_neighbors(self):
        x = self._in.copy()
        x.update(self._out)
        return x

    @property
    def coords(self):
        return (self.left, self.right)

    def add_edge_from(self, c):
        weight = self._in.get(c, 0)
        self._in[c] = weight + 1

    def add_edge_to(self, c):
        weight = self._out.get(c, 0)
        self._out[c] = weight + 1

    def __repr__(self):
        return "ContactState: {}".format(self.coords)


def save_graph(graph, save_path='graph_out.pkl'):
    # graph is l,r discrete state -> ContactState
    new_graph = {}
    for lr_discrete, contact_state in graph.items():
        cs_in, cs_out = {}, {}
        cs_neighbors = {}
        for vn, w in contact_state._in.items():
            cs_in[vn.coords] = w
        for vn, w in contact_state._out.items():
            cs_out[vn.coords] = w
        if contact_state._neighbors:
            for n, w in contact_state._neighbors.items():
                cs_neighbors[n.coords] = w
        new_graph[lr_discrete] = {'in': cs_in, 'out': cs_out, 'neighbors': cs_neighbors}
    pickle.dump(new_graph, open(save_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


def load_graph(save_path='graph_out.pkl'):
    graph = pickle.load(open(save_path, 'rb'))
    new_graph = {}
    for lr_discrete, cs_dict in graph.items():
        cn = ContactState(*lr_discrete, new_graph) # first add all nodes to graph
        new_graph[cn.coords] = cn
    for lr_discrete, cs_dict in graph.items():
        cn._in = {new_graph[x]: y for x,y in cs_dict['in'].items()}
        cn._out = {new_graph[x]: y for x,y in cs_dict['out'].items()}
        cn._neighbors = {new_graph[x]: y for x,y in cs_dict['neighbors'].items()}
    return new_graph


def find_path(start, goal, bfs=True, visited_n=True, classifier=None):
    """
    Args:
        start: starting ContactState node
        goal:  goal ContactState node
        bfs:   bool to choose breadth-first or depth-first search
        visited: bool to use empirical or full graph
    Returns:
        path (if found), or set of visited nodes (if path not found)
    """
    from collections import deque
    node = None
    visited = {start}
    q = deque([[start]])
    classifier = classifier
    if not classifier:
        classifier = lambda s, t: np.random.uniform(0., 1.) < .9
    while node != goal and q:
        path = q.popleft() if bfs else q.pop()  # if bfs FIFO else LIFO queue
        node = path[-1]
        if node == goal:
            break
        if visited_n:
            # if FIFO queue put most visited neighbors at front, else put most visited at back
            node_neighbors = sorted(node.visited_neighbors.keys(),
                                    key=lambda x:node.visited_neighbors[x], reverse=bfs)
        else:
            node_neighbors = sorted(node.neighbors.keys(),
                                    key=lambda x: node.neighbors[x], reverse=bfs)
        for n in node_neighbors:
            if n not in visited and classifier(node, n):
                q.append(path[:] + [n])
                visited.add(n)
    if node != goal:
        print('no path found')
        return []
    else:
        return path


def build_graph(graph=None, csv_path='./original/', add_neighbors=False):
    graph = graph or {}
    dfs = []
    for fn in glob.glob(os.path.join(csv_path, '*.csv')):
        dfs.append(pd.read_csv(fn))

    for df in dfs:
        c = None
        traj = df.values
        disc_contacts = [get_discretized_obj_coord(x) for x in traj]
        for l, r in disc_contacts:
            c_new = graph.get((l, r), ContactState(l, r, graph))
            if (l, r) not in graph:
                graph[c_new.coords] = c_new
            if c and c.coords != c_new.coords:
                c_new.add_edge_from(c)
                c.add_edge_to(c_new)
            #             print(c.coords, '->', c_new.coords)
            c = c_new
        if add_neighbors:
            # constructs neighbors
            for l, r in disc_contacts:
                graph[(l, r)].construct_neighbors()
    return graph


"""
Functions to discretize states
"""

def get_discretized_obj_coord(obs):
    # obs.shape = (13,)
    # Returns the discrete state of left and right contact point: (0 ~ 9) for n_bin=10
    x4 = obs[4];
    y4 = obs[5];
    a4 = obs[8]
    cp_left_x = obs[9];
    cp_right_x = obs[11]
    cp_left_y = obs[10];
    cp_right_y = obs[12]
    l_con, r_con = coordinate_converter([cp_left_x, cp_left_y], [cp_right_x, cp_right_y],
                                             [x4, y4], a4)
    l_cp_discrete = find_discretized_states(l_con, n_bin=10, object_size=200)
    r_cp_discrete = find_discretized_states(r_con, n_bin=10, object_size=200)
    return (l_cp_discrete, r_cp_discrete)


def coordinate_converter(l_contact, r_contact, ob_pos, ob_ori):
    """
    Args:
        l_contact:  [x, y] of left contact
        r_contact:  [x, y] of right contact
        ob_pos:     [x, y] of object marker
        ob_ori:     angle (in radians) of object marker, pointing toward center of object
        offset_len: distance (in pixels) from object marker to center of object
        binwidth:   width of bin to discretize x-axis
        binheight:  heigh tof bin to discretize y-axis
    Returns:
        (l_contact, r_contact) shifted wrt object center and binned
    """
    l_contact, r_contact, ob_pos = np.asarray(l_contact), np.asarray(r_contact), np.asarray(ob_pos)
    y_off = 50 # offset_len # np.cos(ob_ori) * offset_len
    x_off = 50 #offset_len # np.sin(ob_ori) * offset_len
    offset = np.array([x_off, y_off]).T
    rotation_matrix = build_rotation_matrix(-(np.rad2deg(ob_ori) + 135))
    offset_rotated = np.matmul(np.transpose(offset), rotation_matrix)
    ob_offset = ob_pos + offset_rotated
    # ob_offset is the object center
    l_contact_obj_frame_rotated = l_contact - ob_offset
    r_contact_obj_frame_rotated = r_contact - ob_offset

    l_contact_obj_frame = np.matmul(np.transpose(l_contact_obj_frame_rotated), build_rotation_matrix(np.rad2deg(ob_ori) + 270))
    r_contact_obj_frame = np.matmul(np.transpose(r_contact_obj_frame_rotated), build_rotation_matrix(np.rad2deg(ob_ori) + 270))
    return l_contact_obj_frame, r_contact_obj_frame


def find_discretized_states(contact_obj_frame, n_bin, object_size):
    contact_obj_frame_shifted = contact_obj_frame[1] + 100 # taking the y-coordinate and adding 100 so that the y-coordinate >= 0
    if contact_obj_frame_shifted < 0:
        return 0
    elif contact_obj_frame_shifted >= 200:
        return n_bin - 1
    else:
        one_unit = object_size / n_bin
        discrete_state = contact_obj_frame_shifted // one_unit
        # print(discrete_state)
        return discrete_state


def build_rotation_matrix(angle):
    w, h = 2, 2;
    rot_matrix = [[0 for x in range(w)] for y in range(h)]
    radian = np.deg2rad(angle)
    rot_matrix[0][0] = np.cos(radian)
    rot_matrix[0][1] = -np.sin(radian)
    rot_matrix[1][0] = np.sin(radian)
    rot_matrix[1][1] = np.cos(radian)
    return rot_matrix
