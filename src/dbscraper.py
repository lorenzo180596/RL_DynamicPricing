"""

This module contain all the functionality to use real world data to perform the simulation.
A key to use gspread library is needed.

Classes
--------------------
In that file the following classes are defined:

1. DBscraper
   - retrieve a location information based on a spreadsheet imported

Example
-------------
dbs = DBScraper(
        city_name,
        location.latitude,
        location.longitude,
        initial_date,
        final_date
    )

    data = dbs.get_data()

"""

import gspread
from geopy import distance
from datetime import date, timedelta
from property import Competitor


# KEY = [use_your_key]
MAPPING_SS = 'mapping_spreadsheet'
MAX_ALLOWED_DISTANCE = 20               # represent the maximum distance (in km) from centre (all the competitors outside that value are not considered)

class DBScraper():
    def __init__(self, city, latitude, longitude, init_date, final_date):

        self.db = gspread.service_account(KEY)
        self.city = city
        self.city_latitude = latitude
        self.city_longitude = longitude
        self.init_date = init_date
        self.final_date = final_date
        self.competitor_list = None

    def get_data(self):
        
        # get semaphore state
        semaphore = self._check_semaphore()

        if semaphore == 'green':

            # get spreadsheet names in range "init_date" - "final_date"
            spreadsheet_list, date_list = self._get_spreadsheet_list()

            for idx, spreadsheet_name in enumerate(spreadsheet_list):
                if not self.competitor_list:
                    
                    # populate competitor_list with initialized competitors objects
                    self.competitor_list = list()
                    self._init_competitors(spreadsheet_name)

                # update competitors objects
                self._update_competitors(spreadsheet_name, date_list[idx])

            return self.competitor_list

        else:
            print()
            return None

    def _check_semaphore(self):

        # open mapping spreadsheet and get the state of the semaphore
        mapping_spreadsheet = self.db.open(MAPPING_SS)
        semaphore_worksheet = mapping_spreadsheet.worksheet('semaphore')

        return semaphore_worksheet.get('A1')[0][0]

    def _get_spreadsheet_list(self):

        # initialize empty lists  
        spreadsheet_list = list()
        date_list = list()

        # get the names of all the spreadsheets associated with the target city
        mapping_spreadsheet = self.db.open(MAPPING_SS)
        city_worksheet = mapping_spreadsheet.worksheet(self.city)
        date_name_list = city_worksheet.get_all_values()[1::]

        # loop through the spreadsheet names in order to find the ones associated with the target time period
        for element in date_name_list:
            current_date_list = element[0].split('_')
            current_date = date(int(current_date_list[2]), int(current_date_list[1]), int(current_date_list[0]))

            if self.init_date - timedelta(days=6) <= current_date and current_date <= self.final_date:
                spreadsheet_list.append(element[1])
                date_list.append(current_date)
            
        return spreadsheet_list, date_list


    def _init_competitors(self, spreadsheet_name):

        # open the spreadsheet and get the info associated with all the properties
        target_spreadsheet = self.db.open(spreadsheet_name)
        target_worksheet = target_spreadsheet.worksheet('info')
        info_data = target_worksheet.get_all_values()

        # loop through the info in order to initialize competitor objects using the available data
        for info in info_data[1::]:

            #TODO: verify that latitude and longitude parameters exist
            if info[2] and info[3]:
                lat = float(info[2])
                lon = float(info[3])

                #TODO: verify that latitude and longitude parameters are "correct"
                distance_from_centre = distance.distance([lat, lon], [self.city_latitude, self.city_longitude]).kilometers
                if distance_from_centre < MAX_ALLOWED_DISTANCE:

                    if 'stell' in info[9]:
                        stars = int(info[9][0])
                    else:
                        stars = 0

                    # compute total days as it is done in environment.py
                    tot_days = (self.final_date - self.init_date).days + 1 # +1 aggiunto per comprendere il giorno iniziale

                    competitor = Competitor(
                        name=info[1],
                        latitude=lat,
                        longitude=lon,
                        address=info[4],
                        distance_from_centre=distance_from_centre,
                        property_type=info[5].replace('T_', ''),
                        stars=stars,
                        rating_list=list(),
                        rating_cnt_list=list(),
                        pricing_list=list(),
                        use_real_data=True,
                        time_horizon=tot_days,
                    )

                    self.competitor_list.append(competitor)


    def _update_competitors(self, spreadsheet_name, spreadsheet_date):

        # open the spreadsheet and get the info and pricing associated with all the properties
        target_spreadsheet = self.db.open(spreadsheet_name)
        info_worksheet = target_spreadsheet.worksheet('info')
        pricing_worksheet = target_spreadsheet.worksheet('pricing')
        info_data = info_worksheet.get_all_values()
        pricing_data = pricing_worksheet.get_all_values()

        # get property name
        for info in info_data[1::]:
            target_name = info[1]

            # search for the competitor with the correct name and initialize the rating_list and rating_list_cnt attributes
            for competitor in self.competitor_list:
                if competitor.name == target_name:
                    competitor.rating_list.append([spreadsheet_date, float(info[7])])
                    competitor.rating_cnt_list.append([spreadsheet_date, int(info[8])])
                    break

        # get property name
        for pricing in pricing_data[1::]:
            target_name = pricing[1]

            # search for the competitor with the correct name and initialize the pricing_list attribute
            for competitor in self.competitor_list:
                if competitor.name == target_name:
                    
                    # get the price associated with the final_date for each day of the week listed in the spreadsheet
                    for day_step in range(0, 7):

                        current_date = spreadsheet_date + timedelta(days=day_step)
                        if self.init_date <= current_date and current_date <= self.final_date:

                            pricing_list = [current_date]
                            pricing_list_raw = pricing[2 + day_step].strip('][').split('", ')
                            
                            if len(pricing_list_raw) > 1:

                                date_found = False
                                
                                for pricing_raw in pricing_list_raw:
                                    pricing_values = pricing_raw.strip('][').split(', ')
                                    date_list = pricing_values[0].replace('"[\'', '').replace("\'", "").split('-')
                                    _date = date(int(date_list[0]), int(date_list[1]), int(date_list[2]))
                                    
                                    if _date == self.final_date:
                                        _pricing = float(pricing_values[1].replace(']"', '')) / 100
                                        pricing_list.append(_pricing)
                                        competitor.pricing_list.append(pricing_list)

                                        date_found = True
                                        break

                                if not date_found:
                                    pricing_list.append(None)
                                    competitor.pricing_list.append(pricing_list)

                            else:
                                pricing_list.append(None)
                                competitor.pricing_list.append(pricing_list)
                    break


if __name__ == "__main__":

    from geopy.geocoders import Nominatim

    # select the name of the location
    city_name = 'novara'

    # get location geographics info
    geolocator = Nominatim(user_agent="geo_test")
    location = geolocator.geocode(city_name)

    # select initial and final date
    initial_date = date(2023, 3, 15)
    final_date = date(2023, 3, 25)

    dbs = DBScraper(
        city_name,
        location.latitude,
        location.longitude,
        initial_date,
        final_date
    )

    data = dbs.get_data()
    print(data)