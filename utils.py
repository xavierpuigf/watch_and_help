import json
import argparse
from datetime import datetime
import os
import numpy as np
import pdb
import re
import math

def parse_prog(prog):
    program = []
    actions = []
    o1 = []
    o2 = []
    for progstring in prog:
        params = []

        patt_action = r'^\[(\w+)\]'
        patt_params = r'\<(.+?)\>\s*\((.+?)\)'

        action_match = re.search(patt_action, progstring.strip())
        action_string = action_match.group(1).upper()

        param_match = re.search(patt_params, action_match.string[action_match.end(1):])
        while param_match:
            params.append((param_match.group(1), int(param_match.group(2))))
            param_match = re.search(patt_params, param_match.string[param_match.end(2):])

        program.append((action_string, params))
        actions.append(action_string)
        if len(params) > 0:
            o1.append(params[0])
        else:
            o1.append(None)
        if len(params) > 1:
            o2.append(params[1])
        else:
            o2.append(None)

    return actions, o1, o2


def LCS(X, Y):
    # find the length of the strings
    m = len(X)
    n = len(Y)

    # declaring the array for storing the dp values
    L = [[None] * (n + 1) for i in range(m + 1)]

    """Following steps build L[m + 1][n + 1] in bottom up fashion 
    Note: L[i][j] contains length of LCS of X[0..i-1] 
    and Y[0..j-1]"""
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])

                # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1]
    return L[m][n]*1./max(m,n)


# end of function lcs
def computeLCS_multiple(gt_programs, pred_programs):
    lcs_action = []
    lcs_o1 = []
    lcs_o2 = []
    lcs_instr = []
    for it in range(len(gt_programs)):

        lcsa, o1, o2, instr = computeLCS(gt_programs[it], pred_programs[it])
        lcs_action.append(lcsa)
        lcs_o1.append(o1)
        lcs_o2.append(o2)
        lcs_instr.append(instr)
    return np.mean(lcs_action), np.mean(lcs_o1), np.mean(lcs_o2), np.mean(lcs_instr)

def computeLCS(gt_prog, pred_prog):
    if 'stop' in gt_prog[0]:
        stop_index_gt = [it for it, x in enumerate(gt_prog[0]) if x == 'stop'][0]
        gt_prog = [x[:stop_index_gt] for x in gt_prog]

    if 'stop' in pred_prog[0]:
        stop_index_pred = [it for it, x in enumerate(pred_prog[0]) if x == 'stop'][0]
        pred_prog = [x[:stop_index_pred] for x in pred_prog]


    gt_program = list(zip(gt_prog[0], gt_prog[1], gt_prog[2]))
    pred_program = list(zip(pred_prog[0], pred_prog[1], pred_prog[2]))
    action = LCS(gt_prog[0], pred_prog[0])
    obj1 = LCS(gt_prog[1], pred_prog[1])
    obj2 = LCS(gt_prog[2], pred_prog[2])
    instr = LCS(gt_program, pred_program)

    return action, obj1, obj2, instr

class DictObjId:
    def __init__(self, elements=None, include_other=True):
        self.el2id = {}
        self.id2el = []
        self.include_other = include_other
        if include_other:
            self.el2id = {'other': 0}
            self.id2el = ['other']
        if elements:
            for element in elements:
                self.add(element)

    def get_el(self, id):
        if self.include_other and id >= len(self.id2el):
            return self.id2el[0]
        else:
            return self.id2el[id]

    def get_id(self, el):
        el = el.lower()
        if el in self.el2id.keys():
            return self.el2id[el]
        else:
            if self.include_other:
                return 0
            else:
                return self.el2id[el]

    def add(self, el):
        el = el.lower()
        if el not in self.el2id.keys():
            num_elems = len(self.id2el)
            self.el2id[el] = num_elems
            self.id2el.append(el)

    def __len__(self):
        return len(self.id2el)


class Helper:
    def __init__(self, args):
        self.args = args
        self.dir_name = None
        self.setup()

    def setup(self):
        param_name = 'default'
        fname = str(datetime.now())
        if self.args.debug:
            fname = 'debug'
        self.dir_name = '{}/{}/{}'.format(self.args.log_dir, param_name, fname)
        os.makedirs(self.dir_name, exist_ok=True)
        with open('{}/args.txt'.format(self.dir_name), 'w+') as f:
            args_str = str(self.args)
            f.writelines(args_str)


def setup():
    parser = argparse.ArgumentParser(description='RL MultiAgent.')

    # Dataset
    parser.add_argument('--dataset_folder', default='dataset_toy/', type=str) # dataset_subgoals

    # Model params
    parser.add_argument('--action_dim', default=100, type=int)
    parser.add_argument('--object_dim', default=100, type=int)
    parser.add_argument('--relation_dim', default=100, type=int)
    parser.add_argument('--state_dim', default=100, type=int)
    parser.add_argument('--agent_dim', default=100, type=int)
    parser.add_argument('--num_goals', default=3, type=int)

    parser.add_argument('--max_nodes', default=350, type=int)
    parser.add_argument('--max_edges', default=700, type=int)
    parser.add_argument('--max_steps', default=10, type=int)

    # Training params
    parser.add_argument('--num_rollouts', default=5, type=int)
    parser.add_argument('--num_epochs', default=1000, type=int)
    parser.add_argument('--batch_size', default=8, type=int)

    # Logging
    parser.add_argument('--log_dir', default='logdir', type=str)
    parser.add_argument('--print_freq', default=20, type=int)
    parser.add_argument('--debug', action='store_true')


    # Chkpts
    parser.add_argument('--load_path', default=None, type=str)
    args = parser.parse_args()
    helper = Helper(args)
    return helper


def pretty_print_program(program):

    program_joint = list(zip(*program))
    final_instr = [it for it, x in enumerate(program_joint) if x[0] == 'stop']
    if len(final_instr) > 0:
        program_joint = program_joint[:final_instr[0]]
    instructions = []
    for instr in program_joint:
        action, o1, o2 = instr
        o1s = '<{}> ({})'.format(o1[0], o1[1]) if o1[0] not in ['other', 'no_obj', 'stop'] else ''
        o2s = '<{}> ({})'.format(o2[0], o2[1]) if o2[0] not in ['other', 'no_obj', 'stop'] else ''
        instr_str = '[{}] {} {}'.format(action, o1s, o2s)
        instructions.append(instr_str)
    return '\n'.join(instructions)