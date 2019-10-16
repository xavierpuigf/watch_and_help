from utils import DictObjId
import json
import pdb
from torch.utils.data import Dataset, DataLoader


def read_problem(file_problem):
    # This should go in a dataset class
    with open(file_problem, 'r') as f:
        problems = json.load(f)

    problems_dataset = []
    for problem in problems:
        goal_file = problem['file_name']
        graph_file = problem['env_path']
        goal_name = problem['goal']
        program_file = problem['program']

        with open(goal_file, 'r') as f:
            goal_str = f.read()

        with open(program_file, 'r') as f:
            program = f.readlines()
            program = [x.strip() for x in program]


        problems_dataset.append(
            {
                'goal': goal_str,
                'graph_file': graph_file,
                'goal_name': goal_name,
                'program': program
            }
        )
    return problems_dataset


class EnvDataset(Dataset):
    def __init__(self, dataset_file):
        self.dataset_file = dataset_file
        self.problems_dataset = read_problem(dataset_file)

        self.actions = [
            "Walk",  # Same as Run
            # "Find",
            "Sit",
            "StandUp",
            "Grab",
            "Open",
            "Close",
            "PutBack",
            "PutIn",
            "SwitchOn",
            "SwitchOff",
            # "Drink",
            "LookAt",
            "TurnTo",
            # "Wipe",
            # "Run",
            "PutOn",
            "PutOff",
            # "Greet",
            "Drop",  # Same as Release
            # "Read",
            "PointAt",
            "Touch",
            "Lie",
            "PutObjBack",
            "Pour",
            # "Type",
            # "Watch",
            "Push",
            "Pull",
            "Move",
            # "Rinse",
            # "Wash",
            # "Scrub",
            # "Squeeze",
            "PlugIn",
            "PlugOut",
            "Cut",
            # "Eat",
            "Sleep",
            "WakeUp",
            # "Release"
        ]
        self.states = ['on', 'open', 'off', 'closed']
        self.relations = ['inside', 'close', 'facing', 'on']
        self.objects = self.getobjects()

        self.action_dict = DictObjId(self.actions)
        self.object_dict = DictObjId(self.objects)
        self.relation_dict = DictObjId(self.relations)
        self.state_dict = DictObjId(self.states)
        self.num_items = len(self.problems_dataset)

    def __len__(self):
        return self.num_items

    def __getitem__(self, idx):
        problem = self.problems_dataset[idx]
        return problem['graph_file'], problem['goal'], problem['program']



    def getobjects(self):
        print('Getting objects...')
        object_names = []
        for prob in self.problems_dataset:
            with open(prob['graph_file'], 'r') as f:
                graph = json.load(f)
            object_names += [x['class_name'] for x in graph['init_graph']['nodes']]
        object_names = list(set(object_names))
        return object_names
