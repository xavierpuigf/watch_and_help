def distance_reward(graph, target_class="microwave"):
    dist, is_close = self.get_distance(graph, target_class=target_class)

    reward = 0  # - 0.02
    # print(self.prev_dist, dist, reward)
    self.prev_dist = dist
    # is_done = is_close
    is_done = False
    if is_close:
        reward += 0.05  # 1 #5
    info = {'dist': dist, 'done': is_done, 'reward': reward}
    return reward, is_close, info


def reward(visible_ids=None, graph=None):
    '''
    goal format:
    {predicate: number}
    predicate format:
        on_objclass_id
        inside_objclass_id
    '''

    # Low level policy reward

    done = False
    if self.task_type == 'find':
        id2node = {node['id']: node for node in graph['nodes']}
        grabbed_obj = [id2node[edge['to_id']]['class_name'] for edge in graph['edges'] if
                       'HOLDS' in edge['relation_type']]
        print(grabbed_obj)
        reward, is_close, info = self.distance_reward(graph, self.goal_find_spec)
        if visible_ids is not None:

            # Reward if object is seen
            if self.level > 0:
                if len(set(self.goal_find_spec).intersection([node[0] for node in visible_ids])) > 0:
                    reward += 0.5

            if len(set(self.goal_find_spec).intersection(grabbed_obj)) > 0.:
                reward += 1  # 100.
                done = True
        return reward, done, info

    elif self.task_type == 'open':
        raise NotImplementedError
        reward, is_close, info = self.distance_reward(graph, self.goal_find_spec)
        return reward, done, info

    if self.simulator_type == 'unity':
        satisfied, unsatisfied = check_progress(self.get_graph(), self.goal_spec)

    else:
        satisfied, unsatisfied = check_progress(self.env.state, self.goal_spec)

    # print('reward satisfied:', satisfied)
    # print('reward unsatisfied:', unsatisfied)
    # print('reward goal spec:', self.goal_spec)
    count = 0
    done = True
    for key, value in satisfied.items():
        count += min(len(value), self.goal_spec[key])
        if unsatisfied[key] > 0:
            done = False
    return count, done, {}