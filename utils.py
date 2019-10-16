import json
import argparse
from datetime import datetime
import os


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
        tstampname = str(datetime.now())
        self.dir_name = '{}/{}/{}'.format(self.args.log_dir, param_name, tstampname)
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
    parser.add_argument('--num_epochs', default=10, type=int)

    # Logging
    parser.add_argument('--log_dir', default='logdir', type=str)

    args = parser.parse_args()
    helper = Helper(args)
    return helper