# Introduction

This project has been made by Lorenzo Biroli and Christian Borgia and is not updated anymore as of February 2024. It has to be considered only for educational purposes.

Goal of the project was to develop a reinforcement learning algorithm able to dynamically set the price of an hotel room to maximize the profit. At the moment of last update, the algorithm is not able to properly set the correct price and correctly learn an optimized policy.

Project should be upgraded with more refining of the environment simulation and a better definition of the reward of the agent, as well as a proper preprocessing of the data (such as normalization). Different "#TODO" comment can be find along the code that shows different areas of improvement.

The idea behind the use of the reinforcement learning was to get rid of the historical data dependancy, for different reasons:
- historical data may be unsufficient or not present at all (new hotel)
- overfitting risk
- harder setup
- due to the vast numerosity of external economical condition, past data are not a guarantee for current success
- using historical data the algorithm is biased on past performances, and you can not know the customer feedback on the new price
- historical data do not have information about the demand of the period, so you know only the absolute value and not the relative value

## Concept
The reinforcement learning tries to solve the above problem trough a simulation of the performances of an hotel with respect to others competitors.

The simulation is composed by:
1. Creation of the benchmark hotel and competitors hotel
   - Hotels randomly generated with different characteristic of capacity, price, reviews.
2. Arrival of customer
   - customer is randomly generated with different interest and economic availability.
3.  Each customer book the room he likes the most based on his interest and the price
4.  Process is repeated for all the random days that the single epoch can have

During the simulation, a customer can decide to cancel a previous booking, thus making the room available again. Days for each epoch are randomly generated, to increase exploration of different states.

Project allows for the use of real data to be imported in the project to run the simulation. See "dbscraper" module for that.
However, you will need a key in order to use the "gspread" library.

In our case the real data came from a scraping of booking websites such as tripadvisor, performed through a separate program uploaded on the Raspberry PI device. 


# Modules
## src
### Main
File to be run to start the simulation. We can have three different type of simulations:
1. Training -> used to train the neural network. Simulation can be paused and be resumed loading the saved memory buffer and neural networks weights.
2. Validation -> used to validate the algorithm through parallel simulation using the same numpy seed, with and without the application of the dynamic pricing.
3. Test -> used to test single scenarios. To avoid low performances, charts are generated only in this mode.

The type of simulation, as well as all the different option available (early termination, random number of days, random starting, dynamic pricing enabling, numpy seed and so on) can be configured trhough the config file in the config folder.

### Main_sb3
Same as main, where the agent and model comes from the stable baseline package instead of being made from scratch, using pytorch.

### RL - agent
Contains the setup and the methods of the agent, through the pytorch library.

### RL - agent-raylib
Contains the setup and the methods of the agent, through the raylib library.

### RL-model
Contains the setup and the methods of the neural network used by the agent through the DDDQN algorithm.
There are different options and parameter that can be setup for the model, trough the config file, under the "DDDQN" section.

### Customer
Contains the customer class and the definition of all the customers characteristics.

### Property
Contains all the class related to the hotel being generated during the simulations, with all the methods and their characteristics.

### Environment
Contains the setup and all the methods related to the environment. Here are defined all the steps necessary to run properly the simulation, from the generation of the properties and customers, to the booking process.

### Env_settings
Defines the action and state space needed for the DDDQN, as well as the reward.
Different options exists (and more can be added). It is possible to choose the different options to be used through the "config" file, under the section "models".

### Demand
Contains the customer arrival and demand generation.

### Storage
Define all the variables and paths for the saving of the files.

### Dbscraper
Contains the class needed when dealing with real world data. Data can be imported from a spreadsheet and used for the simualtion instead of randomly generated data.

### Lib
Contains the main functions needed for making the simulation work properly.

## rst
Here all the simulation are saved in different folder.
As already stated, charts are found only in the test subfolder.
Please note that in the repository there are already present three different simulations result for educational purpose.
These are not to be intended as working algorithm and they are present just to show what kind of output the code can generate.

## config
Contains the configuration file where the simulation can be setup.
There are a lot of options for the environment including customer demand, customer interests, number of competitors, range of property characteristic, as well as different simulation setup, such as possibility to load previously trained neural networks, different reward structure, early termination and so on.

# RL algorithm
The main algorithm used is the DDDQN, the double dueling deep q-learning algorithm.
Here, the usual q-learning reinforcement learning is enhanced by:
- A neural network instead of the table to estimate the q-values of the state-action pairs.
- An additional target network (hence the term "double") to avoid optimistic q-values.
- A separation in the computation of the q-value between value and advantage, to separate the effect of the action with respect to the state the agent is currently in (hence the term "dueling").

# General Info and setup

## Torch package installation

Go to https://pytorch.org/get-started/locally/ and choose the needed package.
To know if your PC supports cuda (and which CUDA version, go to https://en.wikipedia.org/wiki/CUDA#GPUs_supported).

When installing in the venv environment, if a specific version is required, better to add the command "--no-cache-dir", to avoid using cached version of the package.

If there are network problems that prevent to complete the download, add the following command "--default-timeout=100".

## Package requirements

Activate the venv before installing (or defining the requirements file) any package through the command "venv\Scripts\activate". Then:

- Use the command "python -m pip freeze > requirements.txt" to create the requirements file.
- Use the command "python -m pip install -r requirements.txt" to install the needed packages.

## Code readibility

Use Todo Tree extension to highlight the modification needed in the code: https://thomasventurini.com/articles/the-best-way-to-work-with-todos-in-vscode/

## Toml file

If some configuration has to be set to "True" or "False", write them with lowercase letters. Writing "True" or "False", with starting capital letter will cause error during the loading of the .toml file
