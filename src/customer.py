"""

This module define the customer class used to create the customer instances. Each customer has different characteristic:
- price interest
- position interest
- review interest
- price limit
- minimum rating acceptable
- delete booking probability

Classes
--------------------
In that file the following classes are defined:

1. Customer
   - set up the customer interests and action available


Example
-------------
customer = Customer(price_interest,
        position_interest,
        review_interest,
        price_limit,
        min_value_acceptable,
        delete_booking_prob,
        max_value_for_hotel_score,
        deviation_from_max)

"""

import numpy as np

RATING_INDEX = 0
POSITION_INDEX = 1
PRICE_INDEX = 2
CAPACITY_INDEX = 3

class Customer():
    def __init__(self,
        price_interest,
        position_interest,
        review_interest,
        price_limit,
        min_value_acceptable,
        delete_booking_prob,
        max_value_for_hotel_score,
        deviation_from_max
    ):
        self.price_interest = price_interest
        self.position_interest = position_interest
        self.review_interest = review_interest
        self.price_limit = price_limit
        self.min_value_acceptable = min_value_acceptable
        self.delete_booking_prob = delete_booking_prob
        self.max_value_for_hotel_score = max_value_for_hotel_score
        self.interest_array = np.array([self.review_interest, self.position_interest, self.price_interest])
        self.counter_not_booking = 0
        self.deviation_from_max = deviation_from_max

    def book_property(self, property_info):

        # evaluate properties
        values_perceived = self._evaluate_property(property_info)
        # choose best property
        winner_property_idx = self._choose_property(values_perceived)
        return winner_property_idx

    def get_delete_book(self, day, time_horizon):

        # get random number between 0 and 99.99999999999999 (etc)
        dice_roll = np.random.uniform(0,100)

        # delete booking if dice_roll <= delete probability of customer
        if dice_roll <= self.delete_booking_prob:
            day_cancel_book = np.random.randint(day, time_horizon) 
            return day_cancel_book  
        else:
            return None
  
    def store_cancel_info(self, property, day_cancel_book, day):
        self.property_to_cancel = property
        self.price_paid = property.price[day]
        self.day_cancel_book = day_cancel_book
        
    def increase_comebacks(self):
        self.counter_not_booking += 1

    def _evaluate_property(self, property_info):
        
        property_info[:,PRICE_INDEX] = self.max_value_for_hotel_score - ((property_info[:,PRICE_INDEX])*(self.max_value_for_hotel_score/self.price_limit))
        values_perceived = np.matmul(property_info[:,RATING_INDEX:CAPACITY_INDEX], self.interest_array)
        values_perceived[np.where(property_info[:,PRICE_INDEX]<0)] = 0
        values_perceived[np.where(property_info[:,CAPACITY_INDEX]<1)] = 0
        values_perceived[np.where(values_perceived<self.min_value_acceptable)] = 0

        return values_perceived

    def _choose_property(self, values_perceived):

        max_value = np.max(values_perceived)
        if max_value > 0:
            idx_winner =  np.flatnonzero(values_perceived >= (max_value/100)*self.deviation_from_max)
            return np.random.choice(idx_winner)
        else:
            return None
