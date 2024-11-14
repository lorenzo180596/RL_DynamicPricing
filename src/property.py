from datetime import timedelta
from itertools import count
import numpy as np

RATING_INDEX = 0
POSITION_INDEX = 1
PRICE_INDEX = 2
CAPACITY_INDEX = 3

class Property():
    """Low-level object, parent of Competitor and OurProperty.

    Arguments:
        position (float): value between 0 and 5 representing how much the position of the property is good.
        property_type (str): type of the property.
        stars (int): stars associated with the property.
        rating_init (float): value between 0 and 5 representing the rating of the property at day 1.
        rating_cnt (int): number of customers that have rated the property.
        price_init (int): price of a single room at day 1.
        min_room_price (float): maximum price that the property can set for a single room.
        capacity_init (int): capacity of the property (number of available rooms) at day 1.
        time_horizon (int): number of total days used for the simulation.
        dynamic_price (bool): if True dynamic pricing strategies are used to update the room prices, otherwise the prices remain constants. 
        name (str): name of the property.
        latitude (float): property latitude.
        longitude (float): property longitude.
        address (str): property address.
        distance_from_centre (float): distance in km between the property and the centre of the city.
        rating_list (list): list of lists containing the property rating and the day associated with the data.
        rating_cnt_list (list): list of lists containing the number of customers that have rated the property and the day associated with the data.
        pricing_list (list): list of lists containing the price of the room and the day associated with the data.
        use_real_data (bool): if True real data are used for the property attributes, otherwhise simulated data are used.
    """
    counter = count(0)
    
    def __init__(
        self,
        position=None,
        property_type=None,
        stars=None,
        rating_init=None,
        rating_cnt=None,
        price_init=None,
        min_room_price=None,
        capacity_init=None,
        time_horizon=None,
        dynamic_price=None,
        name=None,
        latitude=None,
        longitude=None,
        address=None,
        distance_from_centre=None,
        rating_list=None,
        rating_cnt_list=None,
        pricing_list=None,
        use_real_data=False
    ):

        self.position = position
        self.property_type = property_type
        self.stars = stars
        self.rating_init = rating_init
        self.rating_cnt = rating_cnt
        self.price_init = price_init
        self.min_room_price = min_room_price
        self.capacity_init = capacity_init
        self.dynamic_price = dynamic_price
        self.name = name
        self.latitude = latitude
        self.longitude = longitude
        self.address = address
        self.distance_from_centre = distance_from_centre
        self.rating_list=rating_list
        self.rating_cnt_list = rating_cnt_list
        self.pricing_list = pricing_list
        self.use_real_data = use_real_data

        #TODO: create time horizon attribute
        if Property.draw_charts:
            self.rating_episodes_cum = [0 for _ in range(time_horizon)]
            self.price_episodes_cum = [0 for _ in range(time_horizon)]
            self.capacity_episodes_cum = [0 for _ in range(time_horizon)]
            self.revenue_episodes_cum = [0 for _ in range(time_horizon)]
            self.how_many_bookings_cum = [0 for _ in range(time_horizon)]
        self.id = next(self.counter)
        
    def _get_position(self):
        pass

    def reset_values(self, time_horizon, price_init, capacity_init, rating_init, new_position):
        self.price_init = price_init
        self.capacity_init = capacity_init
        self.rating_init = rating_init
        self.position = new_position
        self.price = [self.price_init for _ in range(time_horizon)]
        self.rating = [self.rating_init for _ in range(time_horizon)]
        self.capacity = [self.capacity_init for _ in range(time_horizon)]
        self.revenue = [0 for _ in range(time_horizon)]
        self.losses = [0 for _ in range(time_horizon)]
        self.how_many_bookings = [0 for _ in range(time_horizon)]
        Property.info_all_property[self.id][RATING_INDEX] = self.rating_init
        Property.info_all_property[self.id][POSITION_INDEX] = self.position
        Property.info_all_property[self.id][PRICE_INDEX] = self.price_init
        Property.info_all_property[self.id][CAPACITY_INDEX] = self.capacity_init

    def _decrease_price_NoRL(self, rate_decrease_price, day):
        new_price = self.price[day-1] - round(((self.price[day-1]/100)*rate_decrease_price)) 
        if new_price < self.min_room_price:
            self.price[day] = self.min_room_price
        else:
            self.price[day] = new_price
        Property.info_all_property[self.id][PRICE_INDEX] = self.price[day]

    def _increase_price_NoRL(self, rate_increase_price, day):
        self.price[day] = self.price[day-1] + round((self.how_many_bookings[day-1]*(self.price[day-1]/100)*rate_increase_price)) 
        Property.info_all_property[self.id][PRICE_INDEX] = self.price[day]

    def init_new_day(self, day):
        if day > 0:
            self.capacity[day] = self.capacity[day-1]

    def increase_bookings(self, day):
        self.how_many_bookings[day] += 1

    def decrease_capacity(self, day):
        self.capacity[day] -= 1
        Property.info_all_property[self.id][CAPACITY_INDEX] -= 1

    def increase_capacity(self, day):
        self.capacity[day] += 1
        Property.info_all_property[self.id][CAPACITY_INDEX] += 1

    #TODO: add support for real data usage
    def increase_revenue(self, day):
        if not self.use_real_data:
            self.revenue[day] += self.price[day]
        
    def decrease_revenue(self, booked_price, day):
        if not self.use_real_data:
            self.losses[day] += booked_price
        

class Competitor(Property):


    def __init__(
        self,
        position=None,
        property_type=None,
        stars=None,
        rating_init=None,
        rating_cnt=None,
        price_init=None,
        min_room_price=None,
        capacity_init=None,
        time_horizon=None,
        dynamic_price=None,
        rate_increase_price=None,
        rate_decrease_price=None,
        use_real_data=False,
        name=None,
        latitude=None,
        longitude=None,
        address=None,
        distance_from_centre=None,
        rating_list=None,
        rating_cnt_list=None,
        pricing_list=None,
    ):

        super().__init__(
            position=position,
            property_type=property_type,
            stars=stars,
            rating_init=rating_init,
            rating_cnt=rating_cnt,
            price_init=price_init,
            min_room_price=min_room_price,
            capacity_init=capacity_init,
            time_horizon=time_horizon,
            dynamic_price=dynamic_price,
            name=name,
            latitude=latitude,
            longitude=longitude,
            address=address,
            distance_from_centre=distance_from_centre,
            rating_list=rating_list,
            rating_cnt_list=rating_cnt_list,
            pricing_list=pricing_list,
            use_real_data=use_real_data,
        )

        self.rate_decrease_price = rate_decrease_price
        self.rate_increase_price = rate_increase_price


    def update_price(self, day, initial_date = None):
        if self.use_real_data: 
            current_date = initial_date + timedelta(days=day)
            for price_data in self.pricing_list:
                if price_data[0] == current_date:
                    self.price = price_data[1]
                    Property.info_all_property[self.id][PRICE_INDEX] = self.price
                    break
        else:
            if self.dynamic_price == 1:
                if self.how_many_bookings[day-1] == 0:
                    self._decrease_price_NoRL(self.rate_decrease_price, day)
                else:
                    self._increase_price_NoRL(self.rate_increase_price, day)

    def update_rating(self, initial_date, day):
        if self.use_real_data:
            current_date = initial_date + timedelta(days=day)
            for idx in range(1, len(self.rating_list)):
                if self.rating_list[idx - 1] <= current_date and self.rating_list[idx] > current_date:
                    self.rating = self.rating_list[idx-1][1]
                    Property.info_all_property[self.id][RATING_INDEX] = self.rating
                    self.rating_cnt = self.rating_cnt_list[idx-1][1]
                    break
        else:
            # TODO: implement a method for update the rating of competitors when simulated data are used;
            # TODO: update also Property.info_all_property
            pass

class OurProperty(Property):

    def __init__(
        self,
        position=None,
        property_type=None,
        stars=None,
        rating_init=None,
        rating_cnt=None,
        price_init=None,
        capacity_init=None,
        min_room_price=None,
        time_horizon=None,
        dynamic_price = None,
        name=None,
        latitude=None,
        longitude=None,
        address=None,
        rating_cnt_list=None,
        distance_from_centre=None,
        rating_list=None,
    ):

        super().__init__(
            position=position,
            property_type=property_type,
            stars=stars,
            rating_init=rating_init,
            rating_cnt=rating_cnt,
            price_init=price_init,
            min_room_price=min_room_price,
            capacity_init=capacity_init,
            time_horizon=time_horizon,
            dynamic_price = dynamic_price,
            name=name,
            latitude=latitude,
            longitude=longitude,
            address=address,
            distance_from_centre=distance_from_centre,
            rating_list=rating_list,
            rating_cnt_list=rating_cnt_list,
            use_real_data=False,
        )

    def update_rating(self):
        pass

