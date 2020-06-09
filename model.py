"""

Models are ways of representing an instance of a [banana]world.
Models define the rules of engagement. Models create the initial conditions for our world. They create the number of initial agents, their initial parameters, and the schedulers they belong to.

There are two schedulers, one for prodcuts and one for agents. When a model calls step the following happens.
* Go through all of the products (in this case just bananas) and call their step function.
* Go through all of the agents (by type) and do the same. So all growers have their step function called, then all buyers have their step function called.

The below code is v1.0 of BananaWorld. In the future there will be many permutations to the code...

____

Variables which infuence the starting state of the agents within the model are defined in the code below. They are identified as being selected from a range (X,Y) where _x_ is the lowest possible value, and _y_ is the largest. If you want to change these values do so below... just be sure to hae the larger value be higher than the initial one)

"""

from mesa import Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

# local imports
from .agents import BuyerAgent, GrowerAgent, BananaProduct
from .scheduler import RandomActivationByType
import random

# tuples here are mainly expressing floor and ceiling values, respectively
# these values are immutable as per each initialized agent
BUYER_INITIAL_WEALTH = (.5, 5.0)
GROWER_INITIAL_WEALTH = (.5, 5.0)

BUYER_DAILY_INCOME = (.1, 1.0)
GROWER_EXPENSES_COST = (.1, 1.0)

GROWER_SALE_PRICE = (.1, 1.0)
GROWER_SHIPMENT_COST = (.1, 1.0)

GROWER_Co2_GROW_COST = (.1, 1.0)
GROWER_Co2_SHIPMENT_COST = (.1, 1.0)

GROWER_MAX_INVENTORY = (4, 5)
BUYER_MAX_INVENTORY = (2, 3)

BANANA_NUTRIENCE = (.1, 1.0)
BANANA_SHELF_LIFE = (1, 3)

BUYER_HEALTH_INITIAL = (.5, 5.0)
BUYER_HEALTH_DECREASE = (.1, .5)

GRID_STRATEGIES = ["greedy_co2_grid", "greedy_economic_grid", "greedy_health_grid"]


class SimulationACommerce(Model):
    def __init__(self, n=1, gridsize=(1, 1), original_data=None):
        super().__init__()

        if original_data is None:
            original_data = {}
        self.original_data = original_data
        self.num_agents = n * 2
        self.grid = MultiGrid(gridsize[0], gridsize[1], True)
        self.agent_schedule = RandomActivationByType(self)
        self.product_schedule = RandomActivationByType(self)

        # logging formatting
        self.data_collector_agents = DataCollector(model_reporters={
            "Alive Buyers": lambda m: m.agent_schedule.get_type_count_alive(BuyerAgent),
            "Alive Growers": lambda m: m.agent_schedule.get_type_count_alive(GrowerAgent),
            "Carbon Level": lambda m: getattr(m, "co2_current", 0)},
            agent_reporters={
                "id": lambda a: a.unique_id,
                "alive": lambda a: True if a.alive else False,
                "position": lambda a: a.pos,
                "wealth": lambda a: round(a.wealth, 2),
                "health": lambda a: round(a.health, 2),
                "max inventory": lambda a: getattr(a, "max_inventory", None),
                "inventory": lambda a: getattr(a, "inventory", None),
                "usable": lambda a: a.usable_banana_inventory_count(),
                "type": lambda a: "buyer" if isinstance(a, BuyerAgent) else "grower",
                "average inventory age": lambda a: a.average_inventory_age(),
                "inventory count": lambda a: len(a.product()),
                "salary": lambda a: getattr(a, "salary", None),
                "health drop": lambda a: getattr(a, "health_drop", None),
                "product price": lambda a: getattr(a, "product_price", None),
                "product nutrition": lambda a: getattr(a, "nutritional", None),
                "shipment rate": lambda a: getattr(a, "shipment_rate", None),
                "product shelf life": lambda a: getattr(a, "product_shelf_life", None),
                "intelligence": lambda a: getattr(a, "intelligence", None),
                "x pos": lambda a: a.pos[0],
                "y pos": lambda a: a.pos[1],
                "reward": lambda a: getattr(a, "reward", 0, ),
                "last action": lambda a: getattr(a, "last_action", "")
            }
        )
        self.data_collector_product = DataCollector(model_reporters={
            "Bananas Used": lambda m: len(list(filter(lambda a: a.is_used == True, m.product_schedule.agents))),
            "Bananas Trashed": lambda m: len(
                list(filter(lambda a: (a.is_used == False) and a.is_beyond_shelf_life, m.product_schedule.agents)))},
            agent_reporters={
                "shelf life": lambda a: a.shelf_life,
                "is usable": lambda a: a.is_usable,
                "is used": lambda a: a.is_used,
                "owner id": lambda a: a.owner
            }
        )

    def step(self):
        """ Advance agents and products by one step, after logging data """
        self.collect_data()
        self.product_schedule.step()
        self.agent_schedule.step()

    def reset(self):
        """ Reset values of the model """
        for k in self.original_data:
            if getattr(self, k, None):
                setattr(self, k, self.original_data[k])

    def collect_data(self):
        """ Logging agents & products """
        self.schedule = self.agent_schedule
        self.data_collector_agents.collect(self)
        self.schedule = self.product_schedule
        self.data_collector_product.collect(self)


class BananaWorldModel(SimulationACommerce):
    def __init__(self, n=1, random=True, co2=None):
        original_data = {"co2_current": co2}
        super().__init__(n, gridsize=(1, 1), original_data=original_data)
        self.n = n
        self.co2_enabled = False
        self.randomized = random
        if co2:
            self.co2_enabled = True
            self.co2_max = 2 * co2
            self.co2_current = co2
            self.waste_cost = 0.25

        # TODO -- when do trashed bananas stop creating co2
        # TODO -- merge dataframes of overall between two schedulers to show linkages...
        # TODO -- check logging...

    def generate_agents(self):
        # create agents
        for i in range(self.n):
            g = None
            if not self.randomized:
                g = GrowerAgent(unique_id=self.next_id(), model=self, pos=(0, 0), wealth=1.0, product_price=1.0,
                                max_inventory=5)
            else:
                g = GrowerAgent(unique_id=self.next_id(), model=self, pos=(0, 0),
                                wealth=random_range_with_precision(GROWER_INITIAL_WEALTH),
                                product_price=random_range_with_precision(GROWER_SALE_PRICE),
                                max_inventory=random_range_with_precision(GROWER_MAX_INVENTORY, 0),
                                expenses=random_range_with_precision(GROWER_EXPENSES_COST),
                                nutritional=random_range_with_precision(BANANA_NUTRIENCE),
                                shipment_rate=random_range_with_precision(GROWER_SHIPMENT_COST),
                                product_shelf_life=random_range_with_precision(BANANA_SHELF_LIFE, 0),
                                co2_grow=random_range_with_precision(GROWER_Co2_GROW_COST),
                                co2_shipment_rate=random_range_with_precision(GROWER_Co2_SHIPMENT_COST))
            self.grid.place_agent(g, (0, 0))
            self.agent_schedule.add(g)
        for i in range(self.n):
            b = None
            if not self.randomized:
                b = BuyerAgent(unique_id=self.next_id(), model=self, pos=(0, 0), wealth=1.0, max_inventory=2,
                               salary=0.5, strat="greedy_economic")
            else:
                b = BuyerAgent(unique_id=self.next_id(), model=self, pos=(0, 0),
                               wealth=random_range_with_precision(BUYER_INITIAL_WEALTH),
                               max_inventory=random_range_with_precision(BUYER_MAX_INVENTORY, 0),
                               salary=random_range_with_precision(BUYER_DAILY_INCOME),
                               health=random_range_with_precision(BUYER_HEALTH_INITIAL),
                               health_drop=random_range_with_precision(BUYER_HEALTH_DECREASE),
                               strat="greedy_economic")
            self.grid.place_agent(b, (0, 0))
            self.agent_schedule.add(b)

        # collect initial data values
        self.collect_data()

    def step(self):
        super().step()
        if self.co2_enabled is True:
            self.update_co2()

    def update_co2(self):
        trashed_bananas = len(list(filter(lambda a: (a.is_in_trash == True), self.product_schedule.agents)))
        self.increase_co2(trashed_bananas * self.waste_cost)

    def increase_co2(self, i):
        self.co2_current += i

    def decrease_co2(self, d):
        self.co2_current -= d


class BananaWorldModelOnGrid(SimulationACommerce):
    def __init__(self, n=1, gridsize=(3, 3), strats=False, co2=None):
        super().__init__(n, gridsize=gridsize, co2=co2)
        self.strategies = GRID_STRATEGIES if strats else ["greedy_economic_grid"]
        self.n = n

    def generate_agents(self):
        for i in range(self.n):
            g = None
            pos = (random.randint(0, self.grid.width - 1), random.randint(0, self.grid.height - 1))
            if not random:
                g = GrowerAgent(unique_id=self.next_id(), model=self, pos=(3, 3), wealth=1.0, product_price=1.0,
                                max_inventory=5)
            else:
                g = GrowerAgent(unique_id=self.next_id(), model=self, pos=pos,
                                wealth=random_range_with_precision(GROWER_INITIAL_WEALTH),
                                product_price=random_range_with_precision(GROWER_SALE_PRICE),
                                max_inventory=random_range_with_precision(GROWER_MAX_INVENTORY, 0),
                                expenses=random_range_with_precision(GROWER_EXPENSES_COST),
                                nutritional=random_range_with_precision(BANANA_NUTRIENCE),
                                shipment_rate=random_range_with_precision(GROWER_SHIPMENT_COST),
                                product_shelf_life=random_range_with_precision(BANANA_SHELF_LIFE, 0),
                                co2_grow=random_range_with_precision(GROWER_Co2_GROW_COST),
                                co2_shipment_rate=random_range_with_precision(GROWER_Co2_SHIPMENT_COST))
            self.grid.place_agent(g, pos)
            self.agent_schedule.add(g)
        for i in range(self.n):
            b = None
            pos = (random.randint(0, self.grid.width - 1), random.randint(0, self.grid.height - 1))
            if not random:
                b = BuyerAgent(unique_id=self.next_id(), model=self, pos=(0, 0), wealth=1.0, max_inventory=2,
                               salary=0.5)
            else:
                b = BuyerAgent(unique_id=self.next_id(), model=self, pos=pos,
                               wealth=random_range_with_precision(BUYER_INITIAL_WEALTH),
                               max_inventory=random_range_with_precision(BUYER_MAX_INVENTORY),
                               salary=random_range_with_precision(BUYER_DAILY_INCOME),
                               health=random_range_with_precision(BUYER_HEALTH_INITIAL),
                               health_drop=random_range_with_precision(BUYER_HEALTH_DECREASE),
                               strat=random.choice(self.strategies))
            self.grid.place_agent(b, pos)
            self.agent_schedule.add(b)

            # collect initial data values
        self.collect_data()

    def step(self):
        super().step()


class DeepBananaWorld(BananaWorldModelOnGrid):  # see deep_main.py
    def __init__(self, gridsize=(1, 10), buyer_agent=None, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.9,
                 co2=None):
        super().__init__(1, gridsize=gridsize, strats=False, co2=co2)
        self.n = 1
        self.buyer = buyer_agent
        if not self.buyer.unique_id:
            self.buyer.set_id(self.next_id())
        self.buyer.set_model(self)

        self.max_t = max_t
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

    def generate_agents(self):
        g = GrowerAgent(self.next_id(), model=self, pos=(0, 9), expenses=0.0, shipment_rate=1, product_shelf_life=5,
                        max_inventory=8, co2_shipment_rate=1)
        self.grid.place_agent(g, g.pos)
        self.agent_schedule.add(g)
        b = self.buyer
        self.grid.place_agent(b, b.pos)
        self.agent_schedule.add(b)
        self.collect_data()

    def step(self):
        super().step()

    def decay_eps(self):
        self.eps = max(self.eps_end, self.eps_decay * self.eps)
        print(self.eps)


def random_range_with_precision(t, p=2):
    return round(random.uniform(float(t[0]), float(t[1])), p)
