import numpy as np

from gym import spaces
from RL.agent import DDQN_Agent
from math import exp

RATING_INDEX = 0
POSITION_INDEX = 1
PRICE_INDEX = 2
CAPACITY_INDEX = 3

class Env_Settings():

    #######################################################################################
    # GENERAL METHODS THAT CAN BE CALLED BY THE ENVIRONMENT TO DEFINE WHICH METHOD CALL
    #######################################################################################

    def __init__(self, env, config, which_state, which_action, which_agent, which_reward, dynamic_price_competitors):
        self.env = env
        self.config = config
        self.which_state = which_state
        self.which_action = which_action
        self.which_agent = which_agent
        self.which_reward = which_reward
        self.dynamic_price_competitors = dynamic_price_competitors

    def get_agent(self):
        if self.which_agent == "ddqn":
            agent = self._get_agent_ddqn()
        return agent
    
    def get_observation_space(self):
        if self.which_state == "state1":
            observation_space, state_labels = self._get_observation_space1()
            self.discrete = True
        if self.which_state == "state2":
            observation_space, state_labels = self._get_observation_space2()
            self.discrete = True
        if self.which_state == "state3":
            observation_space, state_labels = self._get_observation_space3()
            self.discrete = True
        if self.which_state == "state4":
            observation_space, state_labels = self._get_observation_space4()
            self.discrete = True
        if self.which_state == "state5":
            observation_space, state_labels = self._get_observation_space5()
            self.discrete = True
                
        return observation_space, state_labels
    
    def get_state(self, initial_state = False):
        if self.which_state == "state1":
            state = self._get_state1(initial_state)
        if self.which_state == "state2":
            state = self._get_state2(initial_state)
        if self.which_state == "state3":
            state = self._get_state3(initial_state)
        if self.which_state == "state4":
            state = self._get_state4(initial_state)
        if self.which_state == "state5":
            state = self._get_state5(initial_state)
        return state

    def get_action_space(self):
        if self.which_action == "action1":
            action_space = self._get_action1()
        elif self.which_action == "action2":
            action_space = self._get_action2()
        elif self.which_action == "action3":
            action_space = self._get_action3()
        return action_space

    def update_our_property_price(self, property, info_all_property, action, day):
        if self.which_action == "action1":
            self._compute_our_property_price1(property= property, action= action, day= day)
        if self.which_action == "action2":
            self._compute_our_property_price2(property= property, action= action, day= day)
        if self.which_action == "action3":
            self._compute_our_property_price3(property= property, action= action, day= day)
        info_all_property[property.id][PRICE_INDEX] = property.price[day]

    def get_reward(self):
        if self.which_reward == "reward1":
            reward = self._get_reward1()
        if self.which_reward == "reward2":
            reward = self._get_reward2()
        if self.which_reward == "reward3":
            reward = self._get_reward3()
        if self.which_reward == "reward4":
            reward = self._get_reward4()
        if self.which_reward == "reward5":
            reward = self._get_reward5()
        return reward
    
    #######################################################################################
    # SPECIFIC METHODS THAT DEFINE HOW TO COMPUTE STATE, ACTION AND REWARD, BASED ON 
    # WHAT THE USER WANTS
    #######################################################################################
    
    def _get_agent_ddqn(self):

        agent = DDQN_Agent(n_states= self.env.observation_space.shape[0],
                           n_actions= self.env.action_space.n,
                           batch_size= self.config['ddqn']['batch_size'],
                           hidden_size= self.config['ddqn']['hidden_size'],
                           memory_size= self.config['ddqn']['memory_size'],
                           update_step= self.config['ddqn']['update_step'],
                           learning_rate= self.config['ddqn']['learning_rate'],
                           gamma= self.config['ddqn']['gamma'],
                           tau= self.config['ddqn']['tau']
                       )
        return agent

    def _get_observation_space1(self):

        observation_min = np.array(
            [
                0,                                                      # remaining days
                0,                                                      # our property capacity               
                self.config['our_property']['min_room_price'],          # our property price               
                -500,                                                   # some type of derivative of our property price over time
                0,                                                      # average competitor price       
                -500,                                                   # some type of derivative of mean competitors price over time
                0                                                       # our property day revenue
            ],
            dtype=np.float32,
        )

        observation_max = np.array(
            [
                self.env.tot_days,                                      # remaining days
                self.config['our_property']['max_init_capacity'],       # our property capacity
                1000,                                                   # our property price
                500,                                                    # some type of derivative of our property price over time
                1000,                                                   # average competitor price
                500,                                                    # some type of derivative of mean competitors price over time
                self.config['our_property']['max_init_capacity']*1000,  # our property day revenue
            ],
            dtype=np.float32,
        )

        observation_space = spaces.Box(observation_min, observation_max, dtype=np.float32)
        state_labels = [f"Remaining days", 
                        f"Our propriety capacity", 
                        f"Our propriety price",
                        f"Our propriety price variation",
                        f"Mean competitors price",
                        f"Mean competitors price variation",
                        f"Our propriety day revenue"]

        return observation_space, state_labels
    
    def _get_observation_space2(self):

        observation_min = np.array(
            [
                0,                                                      # remaining days
                0,                                                      # our property capacity               
                self.config['our_property']['min_room_price'],          # our property price               
                0,                                                      # average competitor price
                -200                                                    # last 7 days derivative of mean competitors price
            ],
            dtype=np.float32,
        )

        observation_max = np.array(
            [
                self.env.tot_days,                                      # remaining days
                self.config['our_property']['max_init_capacity'],       # our property capacity #TODO: change max with init capacity
                1000,                                                   # our property price
                1000,                                                   # average competitor price
                200                                                     # last 7 days derivative of mean competitors price
            ],
            dtype=np.float32,
        )

        observation_space = spaces.Box(observation_min, observation_max, dtype=np.float32)
        state_labels = [f"Remaining days", 
                        f"Our propriety capacity", 
                        f"Our propriety price",
                        f"Mean competitors price",
                        f"Last 7 days competitors derivative"]

        return observation_space, state_labels
    
    def _get_observation_space3(self):

        observation_min = np.array(
            [
                0,                                                      # remaining days
                0,                                                      # bookings in the day
                0,                                                      # our property capacity               
                self.config['our_property']['min_room_price'],          # our property price     
                0,                                                      # average competitor price
                -100,                                                   # revenue  moving average of the last 3 days   
                -200                                                    # last 3 days derivative of mean competitors price
            ],
            dtype=np.float32,
        )

        observation_max = np.array(
            [
                self.env.tot_days,                                      # remaining days
                self.config['our_property']['max_init_capacity'],       # bookings in the day
                self.config['our_property']['max_init_capacity'],       # our property capacity 
                1000,                                                   # our property price
                1000,                                                   # average competitor price
                1000,                                                   # revenue moving average of the last 3 days   
                200                                                     # last 3 days derivative of mean competitors price
            ],
            dtype=np.float32,
        )

        observation_space = spaces.Box(observation_min, observation_max, dtype=np.float32)
        state_labels = [f"Remaining days", 
                        f"Bookings",
                        f"Our propriety capacity", 
                        f"Our propriety price",
                        f"Mean competitors price",
                        f"Last 3 days competitors derivative"]

        return observation_space, state_labels
    
    def _get_observation_space4(self):

        observation_min = np.array(
            [
                0,                                                      # remaining days
                0,                                                      # bookings in the day
                0,                                                      # our property capacity               
                self.config['our_property']['min_room_price'],          # our property price   
                0,                                                      # our property rating  
                0,                                                      # our property position  
                0,                                                      # average competitor price
                -100,                                                   # revenue  moving average of the last 3 days   
                -200,                                                   # last 3 days derivative of mean competitors price
                0                                                       # total customers expected

            ],
            dtype=np.float32,
        )

        observation_max = np.array(
            [
                self.env.tot_days,                                      # remaining days
                self.config['our_property']['max_init_capacity'],       # bookings in the day
                self.config['our_property']['max_init_capacity'],       # our property capacity 
                1000,                                                   # our property price
                0,                                                      # our property rating  
                0,                                                      # our property position 
                1000,                                                   # average competitor price
                1000,                                                   # revenue moving average of the last 3 days   
                200,                                                    # last 3 days derivative of mean competitors price
                self.config['customers']['max_number_random']           # total customers expected
            ],
            dtype=np.float32,
        )

        observation_space = spaces.Box(observation_min, observation_max, dtype=np.float32)
        state_labels = ["Remaining days", 
                        "Bookings",
                        "Our propriety capacity", 
                        "Our propriety price",
                        "Our propriety rating",
                        "Our propriety position",
                        "Mean competitors price",
                        "Last 3 days moving average of our propriety revenue",
                        "Last 3 days competitors derivative",
                        "Total customers expected"]

        return observation_space, state_labels

    def _get_state1(self, initial_state):

        if initial_state == True:
            day = self.env.day
            remaining_days = self.env.tot_days - day 
        else:
            day = self.env.day
            remaining_days = self.env.tot_days - day - 1

        # compute mean price of competitors
        competitors_price_sum = 0
        for competitor in self.env.competitors_list:
            competitors_price_sum += competitor.price[day]
        self.env.mean_competitors_price[day] = competitors_price_sum / len(self.env.competitors_list)

        # if step = 0 -> no data on previous prices are availabe, thus their variation is set to 0
        if day > 0:
            our_property_price_variation = self.env.our_property.price[day] - self.env.our_property.price[day-1]
            mean_competitors_price_variation = self.env.mean_competitors_price[day] - self.env.mean_competitors_price[day-1]
        else:
            our_property_price_variation = 0
            mean_competitors_price_variation = 0

        # define current state
        state = np.array(
            [
                remaining_days,
                self.env.our_property.capacity[day],
                self.env.our_property.price[day],
                our_property_price_variation,
                self.env.mean_competitors_price[day],
                mean_competitors_price_variation,
                self.env.our_property.revenue[day]
            ],
            dtype= np.float32
        )

        return state

    def _get_state2(self, initial_state):

        if initial_state == True:
            day = self.env.day
            remaining_days = self.env.tot_days - day 
        else:
            day = self.env.day
            remaining_days = self.env.tot_days - day - 1

        # compute mean price of competitors
        competitors_price_sum = 0
        for competitor in self.env.competitors_list:
            competitors_price_sum += competitor.price[day]
        self.env.mean_competitors_price[day] = competitors_price_sum / len(self.env.competitors_list)

        if self.env.day == 0:
            competitors_price_derivative = 0
        elif self.env.day> 0 and self.env.day <8:
            competitors_price_derivative = (self.env.mean_competitors_price[day] - self.env.mean_competitors_price[0])/day
        else:
            competitors_price_derivative = (self.env.mean_competitors_price[day] - self.env.mean_competitors_price[day-7])/7

        # define current state
        state = np.array(
            [
                remaining_days,
                self.env.our_property.capacity[day],
                self.env.our_property.price[day],
                self.env.mean_competitors_price[day],
                competitors_price_derivative
            ],
            dtype= np.float32
        )

        return state
    
    def _get_state3(self, initial_state):
        
        day = self.env.day

        if initial_state == True:
            self.env.remaining_days = self.env.tot_days - day 
        else: 
            self.env.remaining_days = self.env.tot_days - day - 1

        # compute mean price of competitors
        competitors_price_sum = 0
        for competitor in self.env.competitors_list:
            competitors_price_sum += competitor.price[day]
        self.env.mean_competitors_price[day] = competitors_price_sum / len(self.env.competitors_list)

        if day == 0:
            competitors_price_derivative = 0
            self.env.moving_average_revenue = 0
        elif day> 0 and day <4:
            competitors_price_derivative = (self.env.mean_competitors_price[day] - self.env.mean_competitors_price[0])/day
            self.env.moving_average_revenue = sum(self.env.our_property.revenue[:day])/day
        else:
            competitors_price_derivative = (self.env.mean_competitors_price[day] - self.env.mean_competitors_price[day-3])/3
            self.env.moving_average_revenue = sum(self.env.our_property.revenue[day-3:day])/3

        # define current state
        state = np.array(
            [
                self.env.remaining_days,
                self.env.our_property.how_many_bookings[day],
                self.env.our_property.capacity[day],
                self.env.our_property.price[day],
                self.env.mean_competitors_price[day],
                self.env.moving_average_revenue,
                competitors_price_derivative
            ],
            dtype= np.float32
        )

        return state
    
    def _get_state4(self, initial_state):
        
        day = self.env.day

        if initial_state == True:
            self.env.remaining_days = self.env.tot_days - day 
        else: 
            self.env.remaining_days = self.env.tot_days - day - 1

        # compute mean price of competitors
        competitors_price_sum = 0
        for competitor in self.env.competitors_list:
            competitors_price_sum += competitor.price[day]
        self.env.mean_competitors_price[day] = competitors_price_sum / len(self.env.competitors_list)

        if day == 0:
            competitors_price_derivative = 0
            self.env.moving_average_revenue = 0
        elif day> 0 and day <4:
            competitors_price_derivative = (self.env.mean_competitors_price[day] - self.env.mean_competitors_price[0])/day
            self.env.moving_average_revenue = sum(self.env.our_property.revenue[:day])/day
        else:
            competitors_price_derivative = (self.env.mean_competitors_price[day] - self.env.mean_competitors_price[day-3])/3
            self.env.moving_average_revenue = sum(self.env.our_property.revenue[day-3:day])/3

        # define current state
        state = np.array(
            [
                self.env.remaining_days,
                self.env.our_property.how_many_bookings[day],
                self.env.our_property.capacity[day],
                self.env.our_property.price[day],
                self.env.our_property.rating[day],
                self.env.our_property.position,
                self.env.mean_competitors_price[day],
                self.env.moving_average_revenue,
                competitors_price_derivative,
                self.env.total_customers
            ],
            dtype= np.float32
        )

        return state

    def _get_action1(self):
        
        # Action space:
        # 0 = decrease price by 10%
        # 1 = decrease price by 5 %
        # 2 = do nothing
        # 3 = increase price by 5%
        # 4 = increase price by 10%
        action_space = spaces.Discrete(5) #TODO: evaluate more action to have more tuning
        return action_space
    
    def _get_action2(self):
        # Action space:
        # 0 = decrease price by 20 euros
        # 1 = decrease price by 15 euros
        # 2 = decrease price by 10 euros
        # 3 = decrease price by 5 euros
        # 4 = do nothing
        # 5 = increase price by 5 euros
        # 6 = increase price by 10 euros
        # 7 = increase price by 15 euros
        # 8 = increase price by 20 euros
        action_space = spaces.Discrete(9)
        return action_space
    
    def _get_action3(self):
        # Action space:
        # 0  = decrease price by 30 euros
        # 1  = decrease price by 25 euros
        # 2  = decrease price by 20 euros
        # 3  = decrease price by 15 euros
        # 4  = decrease price by 10 euros
        # 5  = decrease price by 5 euros
        # 6  = do nothing
        # 7  = increase price by 5 euros
        # 8  = increase price by 10 euros
        # 9  = increase price by 15 euros
        # 10 = increase price by 20 euros
        # 11 = increase price by 25 euros
        # 12 = increase price by 30 euros

        action_space = spaces.Discrete(9)
        return action_space

    def _compute_our_property_price1(self, property, action, day):
        if property.dynamic_price:
            # compute new price based on RL agent action
            if day == 0:
                new_price = round(property.price_init - property.price_init*(((action-2)*5)/100))
            else:
                new_price = round(property.price[day-1] - property.price[day-1]*(((action-2)*5)/100))

            # check if new price exceeds previously defined price boundaries, if not, set current price to new price
            if new_price <= property.min_room_price:
                property.price[day] = property.min_room_price    
            else:
                property.price[day] = new_price   
    
    def _compute_our_property_price2(self, property, action, day):
        if property.dynamic_price:
            # compute new price based on RL agent action
            if day == 0:
                new_price = round(property.price_init + (action-4)*5)
            else:
                new_price = round(property.price[day-1] + (action-4)*5)

            # check if new price exceeds previously defined price boundaries, if not, set current price to new price
            if new_price <= property.min_room_price:
                property.price[day] = property.min_room_price    
            else:
                property.price[day] = new_price

    def _compute_our_property_price3(self, property, action, day):
        if property.dynamic_price:
            # compute new price based on RL agent action
            if day == 0:
                new_price = round(property.price_init + (action-6)*5)
            else:
                new_price = round(property.price[day-1] + (action-6)*5)

            # check if new price exceeds previously defined price boundaries, if not, set current price to new price
            if new_price <= property.min_room_price:
                property.price[day] = property.min_room_price    
            else:
                property.price[day] = new_price

    def _get_reward1(self):
        # TODO: valutare analisi statistica reward (residui) distribuzione normale per primi episodi nell'intorno di 0

        # if no bookings in the current timestep
        if self.env.our_property.how_many_bookings[self.env.day] == 0:
            # if there was the possibility of booking
            if self.env.our_property.capacity[self.env.day] > 0:
                # TODO: riferirsi alla capacity rimanente
                # TODO: moltiplicare per prezzo (medio)
                # TODO: inserire reward terminale   
                reward = - self.env.our_property.capacity[self.env.day] / (self.env.tot_days - self.env.day + 1)
            # else if the capacity was 0
            else:
                reward = 0
        # if someone books a room
        else:
            reward = self.env.our_property.how_many_bookings[self.env.day] * (self.env.our_property.price[self.env.day])

        return reward
    
    def _get_reward2(self):
        if self.env.our_property.how_many_bookings[self.env.day] == 0:
            reward = - self.env.our_property.capacity[self.env.day]*self.env.our_property.price[self.env.day]
        else:
            reward = self.env.our_property.how_many_bookings[self.env.day]*self.env.our_property.price[self.env.day]
        return reward
    
    def _get_reward3(self):
        # Reward based on normalization of the percentage increment of the revenue of the day with respect of the moving average of the 
        # revenue of the last three days
        # The reward is computed using an hyperbole function that smooth the reward lowering it the further we are from the deadline (meaning
        # that having an increment of the revenue near the deadline gives you more reward)
        # The parameters are chosen just to have a smooth function between the desired values (-100 and 100)
        # Formula is based on (you can copy the formula below in Desmos)
        # f_{normalizzato}=\frac{200}{1+e^{-\frac{f_{perc}}{20}}}-100 -> this is the revenue normalized between -100 and 100 using the sigmoid function
        # f_{perc}=\frac{f_{fin}-f_{in}}{f_{in}}\cdot100\ -> percentage increment of the revenue 
        # r\left(x\right)\ =\ \frac{f_{normalizzato}}{x}\ \left\{x\ge1\right\} -> this is the reward function

        if self.env.moving_average_revenue == 0:
            percentage_increment = self.env.our_property.how_many_bookings[self.env.day]*10
            if percentage_increment == 0:   
                percentage_increment = -100
            elif percentage_increment > 100:
                percentage_increment = 100
        else:
            percentage_increment = ((self.env.our_property.revenue[self.env.day] - self.env.moving_average_revenue)/self.env.moving_average_revenue)*100
        
        revenue_normalized = (200/(1+exp(-(percentage_increment/20)))) - 100
        if self.env.remaining_days > 0:
            reward = revenue_normalized/self.env.remaining_days
        else:
            reward = revenue_normalized - self.env.our_property.price[self.env.day]*self.env.our_property.capacity[self.env.day]

        return reward

    def _get_reward4(self):
        # Make this reward based on the z-score. 
        # z-score is defined as the number of standard deviations by which the value of a raw score 
        # is above or below the mean value of what is being observed or measured (Wikipedia).
        # It is computed as z = (raw value - mean)/standard deviation;
        # Use this score as the normalized value for the revenue, and then compute the reward using as input this z-score.
        pass

    def _get_reward5(self):

        k_1 = 1
        k_2 = 0.2

        remaining_days = self.env.tot_days - self.env.day

        # y=-k_{1}\left(\frac{c}{x+1}\right)
        if self.env.our_property.how_many_bookings[self.env.day] == 0:
            if self.env.our_property.capacity[self.env.day] > 0:
                reward_1 = - k_1 * (self.env.our_property.capacity[self.env.day] / (remaining_days + 1))
            else:
                reward_1 = 0

            reward = reward_1

        else:
            if self.env.our_property.price[self.env.day] < self.env.mean_competitors_price[self.env.day]:
                # positive reward
                # y=k_{2}\left(x\left(1-\frac{1}{\log\left(\frac{\left(-b_{p}+m_{p}\right)}{m_{p}}\right)^{1}}\right)^{2}\right)
                # res = (- self.env.our_property.price[self.env.day] + self.env.mean_competitors_price[self.env.day]) / self.env.mean_competitors_price[self.env.day]
                # log = np.log10(res)
                reward_2 = k_2 * (self.env.our_property.how_many_bookings[self.env.day] * (
                    1 - 1 / np.log10(
                        (- self.env.our_property.price[self.env.day] + self.env.mean_competitors_price[self.env.day]) / self.env.mean_competitors_price[self.env.day]
                        )
                    ) ** 2
                )

            elif self.env.our_property.price[self.env.day] > self.env.mean_competitors_price[self.env.day]:
                # negative reward
                # y=-k_{2}\left(x\left(1-\frac{1}{\log\left(\frac{\left(b_{p}-m_{p}\right)}{m_{p}}\right)^{1}}\right)^{2}\right)
                # res = (self.env.our_property.price[self.env.day] - self.env.mean_competitors_price[self.env.day]) / self.env.mean_competitors_price[self.env.day]
                # log = np.log10(res)
                reward_2 = - k_2 * (self.env.our_property.how_many_bookings[self.env.day] * (
                    1 - 1 / np.log10(
                        (self.env.our_property.price[self.env.day] - self.env.mean_competitors_price[self.env.day]) / self.env.mean_competitors_price[self.env.day]
                        )
                    ) ** 2
                )

            else:
                reward_2 = k_2

            reward = reward_2

        return reward
    
    