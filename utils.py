import json
import argparse
from datetime import datetime
import os
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

def computeLCS(gt_program, pred_program):
    gt_prog = parse_prog(gt_program)
    pred_prog = parse_prog(pred_program)
    action = LCS(gt_prog[0], pred_prog[0])
    obj1 = LCS(gt_prog[1], pred_prog[1])
    obj2 = LCS(gt_prog[2], pred_prog[2])
    instr = LCS(gt_program, pred_program)
    return action, obj1, obj2, instr

class DictObjId:
    def __init__(self, elements=None):
        self.el2id = {'other': 0}
        self.id2el = ['other']
        if elements:
            for element in elements:
                self.add(element)

    def get_el(self, id):
        if id < len(self.id2el):
            return self.id2el[0]
        else:
            return self.id2el[id]

    def get_id(self, el):
        if el in self.el2id.keys():
            return self.el2id[el]
        else:
            return 0

    def add(self, el):
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
    parser.add_argument('--dataset_file', default='dataset/example_problem.json', type=str)

    # Model params
    parser.add_argument('--action_dim', default=100, type=int)
    parser.add_argument('--object_dim', default=100, type=int)
    parser.add_argument('--relation_dim', default=100, type=int)
    parser.add_argument('--state_dim', default=100, type=int)
    parser.add_argument('--agent_dim', default=100, type=int)

    # Training params
    parser.add_argument('--num_rollouts', default=5, type=int)
    parser.add_argument('--num_epochs', default=100, type=int)

    # Logging
    parser.add_argument('--log_dir', default='logdir', type=str)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--debug', action='store_true')


    # Chkpts
    parser.add_argument('--load_path', default=None, type=str)
    args = parser.parse_args()
    helper = Helper(args)
    return helper