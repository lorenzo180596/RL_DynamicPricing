"""

This module contain all the step necessary to run the booking process for each customer and each hotel.

Classes
--------------------
In that file the following classes are defined:

1. SimulationEnvironment
   - Define all the methods necessary to run the booking process: customer arrival, customer evaluation, customer booking and customer booking cancellation

"""

import numpy as np

from gym import Env

from dbscraper import DBScraper
from demand import Demand
from property import Property, OurProperty, Competitor
from customer import Customer
from env_settings import Env_Settings

# LAT_SCALE = 1 / 111
# LON_SCALE = 1 / 82

RATING_INDEX = 0
POSITION_INDEX = 1
PRICE_INDEX = 2
CAPACITY_INDEX = 3

class SimulationEnvironment(Env):

    def __init__(self, 
                 config, 
                 dynamic_price_competitors, 
                 which_state, 
                 which_action, 
                 which_agent, 
                 which_reward, 
                 dynamic_our_property, 
                 draw_charts):
        
        self.config = config
        self.dynamic_price_competitors = dynamic_price_competitors
        self.env_settings = Env_Settings(env= self, 
                                         config= config, 
                                         which_state= which_state, 
                                         which_action= which_action, 
                                         which_agent= which_agent, 
                                         which_reward= which_reward, 
                                         dynamic_price_competitors= dynamic_price_competitors)
        self.dynamic_our_property = dynamic_our_property
        self.draw_charts = draw_charts

        self.random_starting = self.config["simulation"]["random_starting"]
        self.random_total_days = self.config["simulation"]["random_total_days"]

        # real data flag
        if self.config['simulation']['use_real_data']:
            self.use_real_data = True
            self.location_name = self.config['location']['name']
            self.location_latitude = self.config['location']['latitude']
            self.location_longitude = self.config['location']['longitude']
        else:
            self.use_real_data = False

        self._get_total_days(random= self.random_total_days)

        self.observation_space, self.state_labels = self.env_settings.get_observation_space()
        self.action_space =self.env_settings.get_action_space()

        self.agent = self.env_settings.get_agent()
        
        self.n_hotel = self.config['competitors']['n_hotel']
        self.n_small_hotel = self.config['competitors']['n_small_hotel']
        self.n_bedandbreakfast = self.config['competitors']['n_bedandbreakfast']
        self.n_properties = self.n_hotel + self.n_small_hotel + self.n_bedandbreakfast + 1

        # initialize Property class variables
        Property.info_all_property = np.empty((self.n_properties,4))
        Property.info_all_property.fill(0)
        Property.draw_charts = self.draw_charts
        
        # initialize OUR dynamic property
        self._init_our_property()
        assert self.our_property.id == 0

        # initialize competitors list
        if self.use_real_data:
            self._init_scraped_competitors_list()
        else:
            self._init_random_competitors_list()

        self.all_property_list = [self.our_property] + self.competitors_list
        if self.draw_charts:
            self.cnt_not_booked_customer_cum = [0 for _ in range(self.tot_days)]
            self.daily_arrivals_cum = [0 for _ in range(self.tot_days)]

    def reset(self):
        # Initialize dates and time
        self._init_time()

        if self.random_total_days:
            self._get_total_days(random= True)

        # Reset all values to starting day for a new simulation to begin
        self._reset_new_day_values()

        # Initialize demand and compute arrivals list
        self._init_demand()
        self.demand.compute_customer_arrivals()

        # initialize customers lists (pending and booked) and create a list in which customers
        # that do not book a property in the simulation step will be stored
        self.booked_customers = [[] for _ in range(self.tot_days)]
        self.not_booked_customer = list()
        self.cnt_not_booked_customer = [0 for _ in range(self.tot_days)]
        self.cancelling_customers = [[] for _ in range(self.tot_days)]
        self.pending_customers = [[] for _ in range(self.tot_days)]
        self.mean_competitors_price = [[] for _ in range(self.tot_days)]

        # get current state
        starting_state = self.env_settings.get_state(initial_state = True)

        return starting_state

    def step(self, action):
        
        # create list element for new day
        self._init_new_day_values()
       
        # Create the new customers for the day
        self._get_new_day_customer()  
       
        # update our property price based on RL agent action
        self.env_settings.update_our_property_price(property= self.our_property, info_all_property = Property.info_all_property, action=action, day= self.day)
        
        # update competitors property price and (possibly) attributes
        if self.day > 0:
            self._update_competitors()
        
        # simulate booking process
        self._simulate_booking_process()

        # simulate delete booking process
        self._simulate_delete_booking_process()
    
        # Compute state
        state = self.env_settings.get_state()
        # compute reward
        reward = self.env_settings.get_reward()    
     
        # check terminated state
        if self.day == self.tot_days -1 or (self.config['simulation']['mode'] == 0 and self.config['simulation']['early_termination'] and self.our_property.capacity[self.day] == 0):
            terminated = True
        else:
            terminated = False
            
        self.day += 1
        
        return state, reward, terminated, {}

    def render(self):
        pass

    def close(self):
        pass

    def _get_total_days(self, random = False):
        # get time horizon
        if random:
            min_days = self.config['simulation']['min_days']
            max_days = self.config['simulation']['max_days']
            self.tot_days = np.random.randint(min_days, max_days + 1)
        else:
            self.initial_date = self.config['simulation']['initial_date']
            self.final_date = self.config['simulation']['final_date']
            self.tot_days = (self.final_date - self.initial_date).days + 1 # +1 aggiunto per comprendere il giorno iniziale

    def _init_time(self):
        # initialize elapsed days to 0
        self.day = 0

    def _init_demand(self):
        # get parameters from self.config file
        if self.config['simulation']['random_demand']:
            min_number_random = self.config['customers']['min_number_random']
            max_number_random = self.config['customers']['max_number_random']
            self.total_customers = np.random.randint(min_number_random, max_number_random + 1)
        else:
            self.total_customers = self.config['customers']['total_number_no_random']
        method = self.config['demand']['method']

        # initialize demand object
        self.demand = Demand(self.tot_days, self.total_customers, method)

    def _init_our_property(self):
        # get parameters from self.config file
        property_type = self.config['our_property']['type']
        max_stars = self.config['our_property']['max_stars']
        min_stars = self.config['our_property']['min_stars']
        max_rating_init = self.config['our_property']['max_rating_init']
        min_rating_init = self.config['our_property']['min_rating_init']
        max_rating_cnt = self.config['our_property']['max_rating_cnt']
        min_rating_cnt = self.config['our_property']['min_rating_cnt']
        max_init_price = self.config['our_property']['max_init_price']
        min_init_price = self.config['our_property']['min_init_price']     
        min_room_price = self.config['our_property']['min_room_price']
        max_init_capacity = self.config['our_property']['max_init_capacity']
        min_init_capacity = self.config['our_property']['min_init_capacity']
        min_position = self.config['our_property']['min_position']
        max_position = self.config['our_property']['max_position']

        # get random property attributes within specified boundaries
        stars = np.random.randint(min_stars, max_stars + 1)
        rating_init = round(np.random.uniform(min_rating_init, max_rating_init), 1)
        rating_cnt = np.random.randint(min_rating_cnt, max_rating_cnt + 1)
        price_init = np.random.randint(min_init_price, max_init_price + 1)
        capacity_init = np.random.randint(min_init_capacity, max_init_capacity + 1)
        position = round(np.random.uniform(min_position, max_position), 3)

        # initialize OUR dynamic property object
        self.our_property = OurProperty(position=position,
                                        property_type=property_type,
                                        stars=stars,
                                        rating_init=rating_init,
                                        rating_cnt=rating_cnt,
                                        price_init=price_init,
                                        capacity_init=capacity_init,
                                        min_room_price=min_room_price,
                                        time_horizon= self.tot_days,
                                        dynamic_price= self.dynamic_our_property
                                        )
        
    def _init_random_competitors_list(self):
       
        self.competitors_list = list()

        for cnt_types in range(3):
            if cnt_types == 0:
                type = 'hotel'
                numerosity = self.n_hotel
            elif cnt_types == 1:
                type = 'small_hotel'
                numerosity =self.n_small_hotel
            elif cnt_types == 2:
                type = 'bedandbreakfast'
                numerosity = self.n_bedandbreakfast
            else:
                type = None
                numerosity = 0

            for _ in range(numerosity):
                # get parameters from self.config file
                property_type = self.config[type]['type']
                max_stars = self.config[type]['max_stars']
                min_stars = self.config[type]['min_stars']
                max_rating_init = self.config[type]['max_rating_init']
                min_rating_init = self.config[type]['min_rating_init']
                max_rating_cnt = self.config[type]['max_rating_cnt']
                min_rating_cnt = self.config[type]['min_rating_cnt']
                max_init_price = self.config[type]['max_init_price']
                min_init_price = self.config[type]['min_init_price']
                min_room_price_lb = self.config[type]['min_room_price_lb']
                min_room_price_ub = self.config[type]['min_room_price_ub']
                max_init_capacity = self.config[type]['max_init_capacity']
                min_init_capacity = self.config[type]['min_init_capacity']
                min_position = self.config[type]['min_position']
                max_position = self.config[type]['max_position']
                rate_increase_price = self.config[type]['rate_increase_price']
                rate_decrease_price = self.config[type]['rate_decrease_price']

                # get random property attributes within specified boundaries
                stars = np.random.randint(min_stars, max_stars + 1)
                rating_init = round(np.random.uniform(min_rating_init, max_rating_init), 1)
                rating_cnt = np.random.randint(min_rating_cnt, max_rating_cnt + 1)
                price_init = np.random.randint(min_init_price, max_init_price + 1)
                capacity_init = np.random.randint(min_init_capacity, max_init_capacity + 1)
                position = round(np.random.uniform(min_position, max_position), 3)
                min_room_price = np.random.randint(min_room_price_lb, min_room_price_ub + 1)

                self.competitors_list.append(Competitor(position=position,
                                                        property_type=property_type,
                                                        stars=stars,
                                                        rating_init=rating_init,
                                                        rating_cnt=rating_cnt,
                                                        price_init=price_init,
                                                        min_room_price=min_room_price,
                                                        capacity_init=capacity_init,
                                                        time_horizon= self.tot_days,
                                                        dynamic_price= self.dynamic_price_competitors,
                                                        rate_increase_price= rate_increase_price,
                                                        rate_decrease_price= rate_decrease_price,
                                                        use_real_data= False
                                                        )
                                             )
                
    def _init_scraped_competitors_list(self):

        self.dbscraper = DBScraper(city= self.location_name,
                                   latitude= self.location_latitude,
                                   longitude= self.location_longitude,
                                   init_date= self.initial_date,
                                   final_date= self.final_date,
                                   use_real_data= True
                                   )

        self.competitors_list = self.dbscraper.get_data()

        # the update method here is necessary because when real data are used it fetches the simulate day with the real price of competitors

        self._update_competitors() 

    def _reset_new_day_values(self):
        for property in self.all_property_list:
            if self.random_starting:
                if property.id == 0:
                    min_init_price = self.config['our_property']['min_init_price']   
                    max_init_price = self.config['our_property']['max_init_price'] 
                    min_init_capacity = self.config['our_property']['min_init_capacity']   
                    max_init_capacity = self.config['our_property']['max_init_capacity']  
                    min_rating_init = self.config['our_property']['min_rating_init']     
                    max_rating_init = self.config['our_property']['max_rating_init']
                    min_position = self.config['our_property']['min_position']
                    max_position = self.config['our_property']['max_position']
                else:
                    if property.property_type == 'hotel':
                        min_init_price = self.config['hotel']['min_init_price']
                        max_init_price = self.config['hotel']['max_init_price']
                        min_init_capacity = self.config['hotel']['min_init_capacity']   
                        max_init_capacity = self.config['hotel']['max_init_capacity']
                        min_rating_init = self.config['hotel']['min_rating_init']     
                        max_rating_init = self.config['hotel']['max_rating_init']    
                        min_position = self.config['hotel']['min_position']
                        max_position = self.config['hotel']['max_position']   
                    elif property.property_type == 'small_hotel':
                        min_init_price = self.config['small_hotel']['min_init_price']
                        max_init_price = self.config['small_hotel']['max_init_price']
                        min_init_capacity = self.config['small_hotel']['min_init_capacity']   
                        max_init_capacity = self.config['small_hotel']['max_init_capacity'] 
                        min_rating_init = self.config['small_hotel']['min_rating_init']     
                        max_rating_init = self.config['small_hotel']['max_rating_init']  
                        min_position = self.config['small_hotel']['min_position']
                        max_position = self.config['small_hotel']['max_position']    
                    elif property.property_type == 'bedandbreakfast':
                        min_init_price = self.config['bedandbreakfast']['min_init_price']
                        max_init_price = self.config['bedandbreakfast']['max_init_price']  
                        min_init_capacity = self.config['bedandbreakfast']['min_init_capacity']   
                        max_init_capacity = self.config['bedandbreakfast']['max_init_capacity']  
                        min_rating_init = self.config['bedandbreakfast']['min_rating_init']     
                        max_rating_init = self.config['bedandbreakfast']['max_rating_init'] 
                        min_position = self.config['bedandbreakfast']['min_position']
                        max_position = self.config['bedandbreakfast']['max_position']  
                                        
                new_price_init = np.random.randint(min_init_price, max_init_price + 1)
                new_capacity_init = np.random.randint(min_init_capacity, max_init_capacity + 1)
                new_rating_init = round(np.random.uniform(min_rating_init, max_rating_init), 1)
                new_position = np.random.randint(min_position, max_position + 1)
                property.reset_values(time_horizon= self.tot_days, price_init=new_price_init, capacity_init= new_capacity_init, rating_init= new_rating_init, new_position=new_position)
            else:
                property.reset_values(time_horizon= self.tot_days, price_init=property.price_init, capacity_init= property.capacity_init, rating_init= property.rating_init, new_position= property.position)

    def _init_new_day_values(self):

        for property in self.all_property_list:
            property.init_new_day(self.day)

    def _simulate_booking_process(self):

        # for each customer IN PENDING LIST simulate the booking process
        self._get_pending_customers()

        for customer in self.pending_customers[self.day]:
            info_all_property_tmp = np.copy(Property.info_all_property)
            winner_property_idx = customer.book_property(info_all_property_tmp) 
            
            if winner_property_idx is None:
                self.cnt_not_booked_customer[self.day] += 1
                customer.increase_comebacks()
                if customer.counter_not_booking == 1: 
                    self.not_booked_customer.append(customer)
                elif customer.counter_not_booking > 3:
                    self.not_booked_customer.remove(customer) 

            else:
                self.all_property_list[winner_property_idx].increase_bookings(day = self.day)
                self.all_property_list[winner_property_idx].decrease_capacity(day = self.day)
                self.all_property_list[winner_property_idx].increase_revenue(day = self.day)
                day_cancel_book = customer.get_delete_book(day= self.day, time_horizon= self.tot_days) 
                if day_cancel_book is None:
                    self.booked_customers[self.day].append(customer)
                else:
                    customer.store_cancel_info(self.all_property_list[winner_property_idx], day_cancel_book, self.day)
                    self.cancelling_customers[customer.day_cancel_book].append(customer)

    def _simulate_delete_booking_process(self):
        for customer_cancelling in self.cancelling_customers[self.day]:
            #TODO: add customer cancelling counter
            customer_cancelling.property_to_cancel.increase_capacity(day = self.day)
            customer_cancelling.property_to_cancel.decrease_revenue(booked_price = customer_cancelling.price_paid, day = self.day)

    def _update_competitors(self):

        for competitor in self.competitors_list:
            if competitor.capacity[self.day] > 0:
                if self.use_real_data:
                    competitor.update_price(initial_date= self.initial_date, day= self.day)
                else:
                    competitor.update_price(day= self.day)
            #competitor.update_rating(initial_date= self.initial_date, day= self.day)

    def _get_new_day_customer(self):

        self.day_new_customers_list = list()

        # get parameters from self.config file
        max_price_interest = self.config['customers']['max_price_interest']
        min_price_interest = self.config['customers']['min_price_interest']
        max_position_interest = self.config['customers']['max_position_interest']
        min_position_interest = self.config['customers']['min_position_interest']
        max_review_interest = self.config['customers']['max_review_interest']
        min_review_interest = self.config['customers']['min_review_interest']
        max_price_limit = self.config['customers']['max_price_limit']
        min_price_limit = self.config['customers']['min_price_limit']
        lb_min_value_acceptable = self.config['customers']['lb_min_value_acceptable']
        ub_min_value_acceptable = self.config['customers']['ub_min_value_acceptable']
        max_delete_booking_prob = self.config['customers']['max_delete_booking_prob']
        min_delete_booking_prob = self.config['customers']['min_delete_booking_prob']
        max_value_for_hotel_score = self.config['competitors']['max_value_for_score']
        deviation_from_max = self.config['customers']['deviation_from_max']

        # get the number of customer that arrives in the current day
        current_arrival_customers = self.demand.daily_arrivals[self.day]

        for _ in range(current_arrival_customers):
            # get random customer attributes within specified boundaries
            price_interest = round(np.random.uniform(min_price_interest, max_price_interest), 3)
            position_interest = round(np.random.uniform(min_position_interest, max_position_interest), 3)
            review_interest = round(np.random.uniform(min_review_interest, max_review_interest), 3)
            price_limit = round(np.random.randint(min_price_limit, max_price_limit), 2)
            min_value_acceptable = round(np.random.uniform(lb_min_value_acceptable, ub_min_value_acceptable))
            delete_booking_prob = round(np.random.uniform(min_delete_booking_prob, max_delete_booking_prob), 3)
            
            self.day_new_customers_list.append(
                Customer(
                    price_interest=price_interest,
                    position_interest=position_interest,
                    review_interest=review_interest,
                    price_limit=price_limit,
                    min_value_acceptable=min_value_acceptable,
                    delete_booking_prob=delete_booking_prob,
                    max_value_for_hotel_score=max_value_for_hotel_score,
                    deviation_from_max= deviation_from_max
                )
            )

    def _get_pending_customers(self):
        
        customers_comeback = list(np.random.choice(self.not_booked_customer, np.random.randint(0, min(self.config["simulation"]["max_customers_coming_back_day"], len(self.not_booked_customer) + 1)), replace = False))
        self.pending_customers[self.day] = self.day_new_customers_list + customers_comeback
    

