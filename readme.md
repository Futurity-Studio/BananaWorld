# Banana World

BananaWorld is a suite of Python classes built on top of [Mesa](https://mesa.readthedocs.io/en/master/index.html) and heavily inspired by [DaisyWorld](https://en.wikipedia.org/wiki/Daisyworld).

It was created to help the multiagent simulation community better simulate economic transaction of autonmous agents. The core function of the current codebase is to simulate economic transactions of bananas 🍌

Within the code are the agents *Buyers* and *Growers* which their respective lifecycles. 
Growers grow *Products* (in this case Bananas) and Buyers consume them. Using the existing functions of Mesa, we simulate purchasing cycles of agents as well as lifecycle of products. 

With the example Models provided you can explore, graph, or export as csv, data from worlds with various seed data. This might be as simple as the number of Buyers, or Growers or it might be as complex as the shelf-life of a banana or calcultion of consumer satisfaction.

This library hopes to help those with a basic understanding of OOP in Python to play with the properties defined in the code and discover emergent properties of multiagent simulations handling economic behaviors.

Explore the each class for an explanation of its documentation or simply explore the [colab notebook](https://colab.research.google.com/drive/19D-dua1c1f-83p3PDAtlA2ySKK0wGMG-?usp=sharing)

___

### Changelog:

####v5
Reinforcement learning with DQN
___
* world shrunk to 1 x 10 where buyer learns to move closer to the seller
* see ipynb/colab for more information

####v4
Strategies where buyers can deploy greedy strategies on cost, co2, health values of bananas/sellers
___
* CO2 cost on growers to grow
* CO2 cost on shipping depending on growers 

####v3
Buyers and Sellers are on locations where the buyers have to pay shipping costs _x_ each cell needed to ship product
___
* Grid is now applied randomly
* New features:
    * shipping cost on Growers
* Reformatting of models to allow for inheritance of grid vs non-grid
* Creation of Changelog 
    