"""
The scheduler is used for containing a list of Agents.
The scheduler is used for queueing up, referencing, searching for any time of Agent class (remember anything that extends from Agent class, or a child of Agent is included).

Notable feature on the scheduler:
- each type of agent runs in order
- all Growers are called for step()
- then all Buyers are called for step()
- Agents are randomly selected within their class
- an Agent is only selected for a second time when all other Agents have been selected
"""

from collections import defaultdict
from mesa.time import RandomActivation
from mesa import Model

class RandomActivationByType(RandomActivation):

    def __init__(self, model):
        super().__init__(model)
        # Dictionary to hold agents by their type
        self.agents_by_type = defaultdict(dict)

    def add(self, agent):
        # add agents, get type and add to dictionary
        self._agents[agent.unique_id] = agent
        agent_class = type(agent)
        self.agents_by_type[agent_class][agent.unique_id] = agent

    def remove(self, agent):
        # remove agent, get type and delete to dictionary
        del self._agents[agent.unique_id]
        agent_class = type(agent)
        del self.agents_by_type[agent_class][agent.unique_id]

    def step(self, by_type=True):
        # if by type, all agents of one type before running the next one
        if by_type:
            for agent_class in self.agents_by_type:
                self.step_type(agent_class)
            self.steps += 1
            self.time += 1
        else:
            super().step()

    def step_type(self, object_type):
        # shuffle order and step each type of mesa_agent
        agent_keys = list(self.agents_by_type[object_type].keys())
        self.model.random.shuffle(agent_keys)
        for agent_key in agent_keys:
            self.agents_by_type[object_type][agent_key].step()

    def get_alive_agents(self):
        return list(filter(lambda a: a.alive is True, super().agents))

    def get_type_count(self, type_class):
        return len(self.agents_by_type[type_class].values())

    def get_type_by_attr_value(self, type_class, attr, val):
        return list(filter(lambda a: getattr(a, attr) == val, self.agents_by_type[type_class].values()))

    def get_types(self, object_type):
        return self.agents_by_type[object_type]

    def get_type_count_alive(self, type_class):
        t = self.agents_by_type[type_class].values()
        return len(list(filter(lambda a: a.alive is True, t)))

    def get_type_count_dead(self, type_class):
        return self.get_type_count(type_class) - self.get_type_count_alive(type_class)
