"""
Agent - this is a class that has a unique id, and a step function which runs every cycle.
It is in this step() function where their lifecycle exists.
Things like check life state, buy product, make product, get paid, move to position... all happens here.

Custom Classes which build on Agent:

- BananaWorldAgents - shared information as it pertains to a autonomous agent in our world. They have wealth, life states, inventory, position...
- BananaWorldProducts - shared information as it pertains to products in our world. They have cost, shelf_life usability/expiry sttes, position...

Growers is type of BananaWorldAgents Buyers which is type of BananaWorldAgents.
Bananas is type of BananaWorldProducts. Other product classes can be built as siblings of Bananas.

For more information on each class (Grower, Buyer, Bananas), expand the cell below and check out the code & comments.

Buyer lifecycle is as follows:

- check if the agent is alive (has money > 0). If it is not alive exit the lifecycle
- decrease health
- trash expired Product if any
- consume daily Bananas if any
- earn salary
- check neighbors for Bananas, if there are any which are affordable... buy one (+ shipping cost) if inventory isn't max

Grower lifecycle is as follows:

- check if the agent is alive (has money > 0). If it is not alive exit the lifecycle
- trash expired product if any
- pay expenses
- grow banana if there is one space for it

"""

from mesa import Agent
import numpy as np
import random
import math
from time import time
from scipy.interpolate import interp1d

from .leaner import QNetwork, ReplayBuffer
import torch
import torch.nn.functional as F
import torch.optim as optim

# Constants used for Torch
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 5e-3  # learning rate ...original -> 4
UPDATE_EVERY = 3  # how often to update the network ...original -> 4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Helper methods
def manhattan_distance(x, y):  # grid calculate distance exists but its not manhattan...
    # todo -- contribute to github for toroidal space with manhattan distances
    return sum(abs(a - b) for a, b in zip(x, y))

class AgentACommerce(Agent):
    def __init__(self, unique_id, pos, model, wealth, inventory, max_inventory, health=1,
                 strat="greedy_economic"):
        super().__init__(unique_id, model)
        self.alive = True
        self.wealth = wealth
        self.health = health
        self.max_inventory = max_inventory
        self.inventory = inventory
        self.pos = pos

        self.actions = []
        self.strategy = strat

        self.original_data = {"wealth": wealth, "pos": pos, "health": health}

        # TODO -- how are these values in play? with products, desired ranges?
        #      -- if we fall out do we die? how does it influence satisfaction???
        #      -- need to make a agent & product factory.
        self.satisfaction = 1
        # AGENT METRICS initial / threshold?
        # economic values
        self.cost = 0.0
        # logistical values
        self.logistical = 0.0
        # ecological values
        self.ecological = 0.0
        # health values
        self.health = health
        # social values
        self.social = 0.0

    def set_id(self, unique_id):
        self.unique_id = unique_id

    def set_model(self, model):
        self.model = model

    def is_alive(self):
        if self.health <= 0:
            self.alive = False
        return True if self.alive == 1 else False

    def die(self):
        self.alive = False

    def add_wealth(self, val):
        self.wealth += val

    def remove_wealth(self, val):
        self.wealth -= val

    def decrease_health(self, d):
        self.health -= d

    def increase_health(self, d):
        self.health = np.clip(self.health + d, 0, self.original_data["health"])

    def step(self):
        super().step()
        if self.wealth <= 0 or self.health <= 0:
            self.die()

    def decrement_inventory(self):
        self.inventory -= 1

    def increment_inventory(self):
        self.inventory += 1

    def trash_expired_or_used_product(self):
        product = list(filter(lambda x: x.owner == self.unique_id, self.model.product_schedule.agents))
        for p in product:
            if p.is_beyond_shelf_life or (not p.is_usable):
                p.remove()
                p.trash()
                self.decrement_inventory()

    def average_inventory_age(self):
        product = list(filter(lambda x: x.owner == self.unique_id, self.model.product_schedule.agents))
        return sum(p.age for p in product) / len(product) if len(product) > 0 else 0

    def product(self):
        return list(filter(lambda x: x.owner == self.unique_id, self.model.product_schedule.agents))

    def reset(self):
        for k in self.original_data:
            if getattr(self, k, None):
                setattr(self, k, self.original_data[k])
        self.alive = True

    # Shared Banana Helpers

    def usable_banana_inventory_count(self):
        return len(self.usable_banana_inventory())

    def usable_banana_inventory(self):
        return list(filter(lambda x: x.owner == self.unique_id and x.is_usable == True, self.model.product_schedule.agents))


    # Utility Helpers

    def calculate_shipment_cost(self, buyer, seller):
        return manhattan_distance(buyer.pos, seller.pos) * seller.shipment_rate

    def calculate_shipment_co2(self, buyer, seller):
        d = self.get_distance(buyer.pos, seller.pos)
        return seller.co2_grow_cost + d * seller.co2_shipment_rate

    def get_distance(self, pos_1, pos_2):
        x1, y1 = pos_1
        x2, y2 = pos_2

        dx = np.abs(x1 - x2)
        dy = np.abs(y1 - y2)
        if self.model.grid.torus:
            dx = min(dx, self.model.grid.width - dx)
            dy = min(dy, self.model.grid.height - dy)
        return np.sqrt(dx * dx + dy * dy)


class BuyerAgent(AgentACommerce):
    # todo - need to setup thresholds for agent actions

    def __init__(self, unique_id, model, pos, strat, state_size=1, wealth=1.0, max_inventory=1,
                 salary=1.0, health=1.0, health_drop=1, intelligence=0):
        super().__init__(unique_id, pos, model, wealth, inventory=0, max_inventory=max_inventory,
                         health=health, strat=strat)

        self.salary = salary
        self.health_drop = health_drop
        self.intelligence = intelligence
        self.initialize_strategy()

        # RL below~~~~~~~~~~~~~~~~
        # initial vars
        self.state_size = state_size
        self.action_size = len(self.actions)
        # self.seed = random.seed(seed)
        seed = int(time())

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, self.action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, self.action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(self.action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        # RL above~~~~~~~~~~~~~~~~

        self.original_data.update({"health_drop": self.health_drop, "intelligence": self.intelligence})

        # tracking below
        self.reward = 0
        self.last_action = "_"

    def reset(self):
        super().reset()
        for k in self.original_data:
            if getattr(self, k, None):
                setattr(self, k, self.original_data[k])

    def initialize_strategy(self):
        if self.intelligence == 0:
            self.actions.extend([
                self.step_inventory,
                self.step_consumables,
                self.step_lifecycle_transaction,
                self.step_agent_transactions])
        else:
            self.actions.extend([
                self.step_consumables,
                self.step_agent_transactions,
                self.move_down,
                self.move_up
                # self.move_right,
                # self.move_left
            ])

    def step(self):
        # print("Buyer... updating\n")
        # ~~ LIFECYCLE ~~
        # check if actionable
        super().step()
        if not self.is_alive():
            return
        self.decrease_health(self.health_drop)
        # ~~ /LIFECYCLE ~~

        if self.intelligence > 0:
            self.intelligent_step()
            # self.learn_step()

            #  learn
            # update values...
            # update actions
            # self.do_learned_actions()
        else:
            for a in self.actions:
                a()
        # print("Buyer... done updating")

    # the below sections are actions
    def step_inventory(self):
        super().trash_expired_or_used_product()

    def step_consumables(self):  # TODO -- CHANGE THIS BACK TO NORMAL!
        # consume banana
        if self.usable_banana_inventory_count():
            self.consume_daily_banana()

    def step_lifecycle_transaction(self):
        self.earn_salary()

    def step_agent_transactions(self):
        grid = True if "grid" in self.strategy else False
        econ = True if "econo" in self.strategy else False
        ecol = True if "co2" in self.strategy else False
        health = True if "health" in self.strategy else False
        greed = True if "greed" in self.strategy else False

        if econ:
            self.try_buy_cost(greedily=greed, grid=grid)
        elif health:
            self.try_buy_nutrients(greedily=greed, grid=grid)
        elif ecol:
            self.try_buy_co2(greedily=greed, grid=grid)
        else:
            raise Exception(f'invalid strategy given: {self.strategy}')

    # the above sections are actions

    def try_buy_cost(self, greedily=False, grid=False):
        # check for bananas on my location
        if self.inventory < self.max_inventory:
            if grid:
                purchasable_bananas = self.select_affordable_growers(self.growers_in_grid_with_bananas())
            else:
                purchasable_bananas = self.select_affordable_growers(self.growers_in_local_with_bananas())

            if purchasable_bananas and greedily:
                self.buy_banana_greedily(purchasable_bananas, "cost")
            elif purchasable_bananas and not greedily:
                self.buy_banana_randomly(purchasable_bananas)

    def try_buy_nutrients(self, greedily=False, grid=False):
        if self.inventory < self.max_inventory:
            if grid:
                purchasable_bananas = self.select_affordable_growers(self.growers_in_grid_with_bananas())
            else:
                purchasable_bananas = self.select_affordable_growers(self.growers_in_local_with_bananas())

            if purchasable_bananas and greedily:
                self.buy_banana_greedily(purchasable_bananas, "health")
            elif purchasable_bananas and not greedily:
                self.buy_banana_randomly(purchasable_bananas)

    def try_buy_co2(self, greedily=False, grid=False):
        if self.inventory < self.max_inventory:
            if grid:
                purchasable_bananas = self.select_affordable_growers(self.growers_in_grid_with_bananas())
            else:
                purchasable_bananas = self.select_affordable_growers(self.growers_in_local_with_bananas())

            if purchasable_bananas and greedily:
                self.buy_banana_greedily(purchasable_bananas, "co2")
            elif purchasable_bananas and not greedily:
                self.buy_banana_randomly(purchasable_bananas)

    #   banana selection below

    def buy_banana_randomly(self, growers):
        random_seller = self.random.choice(growers)
        banana = random_seller.get_banana_for_sale()

        # transaction here
        self.transact_banana(random_seller, banana)

    def buy_banana_greedily(self, growers, attr):
        if attr == "cost":
            sorted_growers = sorted(growers,
                                    key=lambda a: getattr(a.get_banana_for_sale(attribute=attr, desc=True),
                                                          attr) + self.calculate_shipment_cost(self, a))
            seller = sorted_growers[0]
            banana = seller.get_banana_for_sale(attribute=attr, desc=True)
        elif attr == "health":
            sorted_growers = sorted(growers,
                                    key=lambda a: getattr(a.get_banana_for_sale(attribute=attr, desc=True), attr),
                                    reverse=True)
            seller = sorted_growers[0]
            banana = seller.get_banana_for_sale(attribute=attr, desc=True)
        elif attr == "co2":
            sorted_growers = sorted(growers,
                                    key=lambda a: a.co2_grow_cost + self.calculate_shipment_co2(self, a))
            seller = sorted_growers[0]
            banana = seller.get_banana_for_sale()
        self.transact_banana(seller, banana)

    def transact_banana(self, seller, banana):
        total_cost = banana.cost + self.calculate_shipment_cost(self, seller)
        if self.model.co2_enabled:
            self.model.increase_co2(self.calculate_shipment_co2(self, seller))
        self.remove_wealth(total_cost)
        seller.add_wealth(total_cost)
        banana.set_owner(self)
        seller.decrement_inventory()
        self.increment_inventory()

    def select_affordable_growers(self, growers):
        return list(filter(lambda x: x.product_price <= self.wealth, growers))

    def select_affordable_growers_with_shipment_cost(self, growers):
        return list(filter(lambda x: (x.product_price + self.calculate_shipment_cost(self, x)) <= self.wealth, growers))

    def growers_in_local_with_bananas(self):
        this_cell = self.model.grid.get_cell_list_contents([self.pos])
        growers = [obj for obj in this_cell if isinstance(obj, GrowerAgent)]
        growers_with_supply = list(filter(lambda x: x.inventory >= 1, growers))
        return growers_with_supply

    # this introduces a buy limitation
    # TODO - doesnt seem to be called by anything...
    def select_growers_in_neighborhood_who_have_bananas(self, radius=1):
        # grid size is 1,1... select random grower from neighbor which has bananas
        # print(self.pos)
        this_cell = self.model.grid.get_cell_list_contents([self.pos])
        growers = [obj for obj in this_cell if isinstance(obj, GrowerAgent)]
        growers_with_supply = list(filter(lambda x: x.inventory >= 1, growers))
        return growers_with_supply

    def growers_in_grid_with_bananas(self):
        agents = self.model.grid.get_neighbors(self.pos, True, True, max(self.model.grid.width, self.model.grid.height))
        growers = [obj for obj in agents if isinstance(obj, GrowerAgent)]
        growers_with_supply = list(filter(lambda x: x.inventory >= 1, growers))
        return growers_with_supply

    # ~~~~~~ end of transaction content here ~~~~~~

    def earn_salary(self):
        self.wealth += self.salary

    def consume_daily_banana(self, oldest_first=True):
        bananas = self.usable_banana_inventory()
        banana_to_eat = None
        if oldest_first:
            banana_to_eat = sorted(bananas, key=lambda x: x.shelf_life)[0]
        else:
            banana_to_eat = self.random.choice(bananas)
        banana_to_eat.use()
        self.increase_health(banana_to_eat.health)

    #   movement actions below
    # TODO -- add toroidal adjust before or after move_agent
    def move_down(self):
        next_moves = self.model.grid.get_neighborhood(self.pos, False, False)
        # print(next_moves)
        next_move = \
            list(filter(lambda a: (a == self.model.grid.torus_adj((self.pos[0], (self.pos[1] - 1)))), next_moves))[0]
        self.model.grid.move_agent(self, self.model.grid.torus_adj(next_move))

    def move_up(self):
        next_moves = self.model.grid.get_neighborhood(self.pos, False, False)
        # print(next_moves)
        next_move = \
            list(filter(lambda a: (a == self.model.grid.torus_adj((self.pos[0], (self.pos[1] + 1)))), next_moves))[0]
        self.model.grid.move_agent(self, self.model.grid.torus_adj(next_move))

    def move_left(self):
        next_moves = self.model.grid.get_neighborhood(self.pos, False, False)
        # print(next_moves)
        next_move = \
            list(filter(lambda a: (a == self.model.grid.torus_adj(((self.pos[0] + 1), self.pos[1]))), next_moves))[0]
        self.model.grid.move_agent(self, self.model.grid.torus_adj(next_move))

    def move_right(self):
        next_moves = self.model.grid.get_neighborhood(self.pos, False, False)
        # print(next_moves)
        next_move = \
            list(filter(lambda a: (a == self.model.grid.torus_adj(((self.pos[0] - 1), self.pos[1]))), next_moves))[0]
        self.model.grid.move_agent(self, self.model.grid.torus_adj(next_move))

    # ~~~~~~ end of transaction content here ~~~~~~

    #  DQN - reinforcement learning here
    def learn_step(self, state, action, reward, next_state, done):
        # store memory into replay memory
        self.memory.add(state, action, reward, next_state, done)

        # learn every 'UPDATE_EVERY' # of time steps
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def do_learned_actions(self):
        a = random.choice(self.actions)
        a()

    def calculate_state(self):
        """This function calcultes state as it relates to
        both agent and the world the agent is it in
        :return:
            (agent-x-pos, agent-pos-y, distance_from_seller, agent health, agent wealth, agent-inventory/max_inventory, c02 level, # of useable bananas)
        """
        agents = self.model.grid.get_neighbors(self.pos, True, True, max(self.model.grid.width, self.model.grid.height))
        growers = [obj for obj in agents if isinstance(obj, GrowerAgent)]
        grower = growers[0]
        d = manhattan_distance(self.pos, grower.pos)
        return np.asarray([self.pos[0], self.pos[1], d, self.health, self.wealth, self.max_inventory - self.inventory,
                           self.model.co2_current, self.usable_banana_inventory_count()],
                          dtype=np.float32)

    def calculate_reward(self):
        # logistics
        agents = self.model.grid.get_neighbors(self.pos, True, True, max(self.model.grid.width, self.model.grid.height))
        growers = [obj for obj in agents if isinstance(obj, GrowerAgent)]
        grower = growers[0]
        d = manhattan_distance(self.pos, grower.pos)
        d_reward = pow(2, (-1 * d))

        # health
        h_reward = math.sin(0.5 * (self.health / self.original_data["health"]) * math.pi)

        # money
        m_reward = math.sin(0.5 * (self.wealth / self.original_data["wealth"]) * math.pi)

        # ecological
        e_reward = 1 - (abs(self.model.co2_max - self.model.co2_current) / self.model.co2_max)

        reward = (d_reward * 0.15) + (h_reward * 0.3) + (m_reward * 0.25) + (e_reward * .3)
        self.reward = reward
        return self.reward

    def intelligent_step(self):
        state = self.calculate_state()
        action = self.act(state, self.model.eps)  # calculate action
        # print(action)
        self.last_action = action
        # print(self.actions[action])
        self.actions[action]()
        next_state = self.calculate_state()
        reward = self.calculate_reward()
        # todo--- a thing to note... next state isnt always deterministic.. since the buyer can do different tasks
        # also -- done parameter is always false because we check if the agent is alive earlier
        self.learn_step(state, action, reward, next_state, done=False)


class GrowerAgent(AgentACommerce):
    # todo -- think about negotiation.. right now they do not dictate anything just the buyers
    # see get_bananas_for_sale func

    def __init__(self, unique_id, model, pos, wealth=1, product_price=1.0, max_inventory=1, expenses=1.0,
                 nutritional=1.0, shipment_rate=0, product_shelf_life=1, co2_grow=0, co2_shipment_rate=0):
        super().__init__(unique_id, pos, model, wealth, inventory=0, max_inventory=max_inventory)
        self.pos = pos
        self.product_price = product_price
        self.expenses = expenses
        self.nutritional = nutritional
        self.shipment_rate = shipment_rate
        self.product_shelf_life = product_shelf_life
        self.co2_grow_cost = co2_grow
        self.co2_shipment_rate = co2_shipment_rate

    def step(self):
        # print("Grower... updating\n")

        super().step()
        if not self.is_alive():
            return
        super().trash_expired_or_used_product()

        if self.model.co2_enabled:
            self.affect_co2()
        self.pay_expenses()

        # if inventory isn't full, grow bananas
        if self.inventory < self.max_inventory:
            self.grow_banana()
            # print(self.__dict__)
        # print("\nGrower... done updating")

    def grow_banana(self):  # TODO - handle when cost of banana is different from  seller...
        b = BananaProduct(unique_id=self.model.next_id(), model=self.model, pos=self.pos,
                          shelf_life=self.product_shelf_life, cost=self.product_price, health=self.nutritional)
        b.set_owner(self)
        self.inventory += 1
        self.model.product_schedule.add(b)
        self.model.grid.place_agent(b, self.pos)

    def get_banana_for_sale(self, attribute="shelf_life", order=True, desc=False):
        bananas = list(filter(lambda x: x.owner == self.unique_id, self.model.product_schedule.agents))
        if order:
            return sorted(bananas, key=lambda x: getattr(x, attribute), reverse=desc)[0]
        else:
            return bananas[0]

    def pay_expenses(self):
        self.wealth -= self.expenses

    # this function reduces co2 according to how many bananas are in inventory... bananas trees reduce co2
    # currently they decrease co2 with a value of 1...
    # TODO - this value should CHANGE from an arbitrary one to a stat one
    def affect_co2(self):
        bananas = list(filter(lambda x: x.owner == self.unique_id, self.model.product_schedule.agents))
        for b in bananas:
            self.model.decrease_co2(1)


# Products below ---


class ProductACommerce(Agent):
    grid = None # TODO -- check if this matters at all

    def __init__(self, unique_id, pos, model, shelf_life=-1, cost=1.0, health=1.0, logi=1.0, eco=1.0, soc=1.0):
        super().__init__(unique_id, model)
        self.pos = pos

        # PRODUCT METRICS
        """
            The below metrics are used to describe the categorical values of a product as it relates to a consumer
        """
        # economic values
        self.cost = cost
        # logistical values
        self.logistical = logi
        # ecological values
        self.ecological = eco
        # health values
        self.health = health
        # social values
        self.social = soc

        # lifecycle
        self.shelf_life = shelf_life
        self.age = 0
        self.is_used = False
        self.is_usable = True
        self.is_beyond_shelf_life = False
        self.owner = None
        self.is_in_trash = False

    def set_owner(self, owner):
        if isinstance(owner, Agent):
            self.owner = owner.unique_id
        else:
            self.owner = owner

    def use(self):
        self.is_used = True
        self.prepare_discard(old=False)

    def prepare_discard(self, old):
        if old:
            self.is_beyond_shelf_life = True
            self.is_usable = False
        else:
            self.is_usable = False

    def trash(self):
        self.is_in_trash = True

    def check_expired(self):
        if self.shelf_life == -1:  # setting lifespan to -1 makes the item nonperishable
            return False
        elif self.shelf_life == 0:
            return True
        else:
            return False

    def increment_age(self):
        if self.shelf_life == -1 or self.shelf_life == 0:
            return False
        else:
            self.shelf_life -= 1
            self.age += 1
            return True

    def remove(self):
        self.model.grid.remove_agent(self)
        # for reporting purposes... we should not remove products from schedulers
        self.set_owner(None)


class BananaProduct(ProductACommerce):
    def __init__(self, unique_id, model, pos, shelf_life=0, cost=1, health=1.0):
        super().__init__(unique_id, pos, model, shelf_life, cost, health=health)

    def step(self):
        # print("Banana... updating\n")
        if not self.is_usable:
            return
        if super().check_expired():
            super().prepare_discard(old=True)
        super().increment_age()
        # print(self.__dict__)
        # print("Banana... done updating\n")
