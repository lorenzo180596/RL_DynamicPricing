import os
import sys
import toml
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from datetime import datetime
from operator import add 
from itertools import accumulate
from matplotlib import pyplot as plt
from environment import SimulationEnvironment



def get_config_file():
    path_to_look_in = os.path.abspath('')
    config_path = None
    name = 'config.toml'
    for root, dirs, files in os.walk(path_to_look_in):
        if name in files:
            config_path = os.path.join(root, name)
    if config_path is None:
        raise "Configuration file not found"
    else:    
        print("Found the configuration file in: ", config_path)
        return config_path

def compute_cumulative_data_episodes(env):
    for property in env.all_property_list:
            property.price_episodes_cum = list(map(add, property.price_episodes_cum, property.price))
            property.capacity_episodes_cum = list(map(add, property.capacity_episodes_cum, property.capacity))
            property.revenue_episodes_cum = list(map(add, property.revenue_episodes_cum, property.revenue))
            property.how_many_bookings_cum = list(map(add, property.how_many_bookings_cum, property.how_many_bookings))
    env.cnt_not_booked_customer_cum = list(map(add, env.cnt_not_booked_customer_cum, env.cnt_not_booked_customer))
    env.daily_arrivals_cum = list(map(add, env.daily_arrivals_cum, env.demand.daily_arrivals))

def compute_mean_data_episodes(env, episode):
    for property in env.all_property_list:
            property.price_episodes_mean = [day_cnt /(episode+1) for day_cnt in  property.price_episodes_cum]
            property.capacity_episodes_mean = [day_cnt /(episode+1) for day_cnt in  property.capacity_episodes_cum]
            property.revenue_episodes_mean = [day_cnt /(episode+1) for day_cnt in  property.revenue_episodes_cum]
            property.how_many_bookings_mean = [day_cnt /(episode+1) for day_cnt in  property.how_many_bookings_cum]
    env.cnt_not_booked_customer_mean = [day_cnt /(episode+1) for day_cnt in  env.cnt_not_booked_customer_cum]
    env.daily_arrivals_mean = [day_cnt /(episode+1) for day_cnt in env.daily_arrivals_cum]


def get_x_ticks():
    resolution = (env.tot_days//50)*2 + 2
    tot_days_ticks = env.tot_days + (resolution - (env.tot_days % resolution))
    x_ticks = [cnt_ticks*resolution for cnt_ticks in range(int(tot_days_ticks/resolution))]  
    return x_ticks

def get_y_ticks(max_y_reached):
    resolution = (max_y_reached//50)*5 + 5
    max_y_reached = max_y_reached + (resolution - (max_y_reached % resolution))
    y_ticks = [cnt_ticks*resolution for cnt_ticks in range(int(max_y_reached/resolution)+1)]
    return y_ticks

def plot_data(env, output_path):

    output_folderpath = os.path.join(output_path, "plots")
    os.mkdir(output_folderpath)

    x_ticks = get_x_ticks()

    n_property = len(env.all_property_list)
    x_days = [day for day in range(env.tot_days)]
    colors = plt.cm.hsv(np.linspace(0, 0.9, n_property + 1))

    ####################################################
    # Chart 1 - Daily arrivals
    ####################################################

    plt.figure(1, figsize= (10,6))
    plt.title("Mean daily arrivals")
    max_y_reached = max(env.daily_arrivals_mean)
    plt.plot(env.daily_arrivals_mean)
    plt.ylabel("Daily mean new customer")
    plt.grid()

    y_ticks = get_y_ticks(max_y_reached)
    plt.yticks(y_ticks)
    plt.xticks(x_ticks)
    plt.xlabel('Days')

    plt.savefig(os.path.join(output_folderpath, 'Mean daily arrivals.png'))


    ####################################################
    # Chart 2 - Mean capacity
    ####################################################

    plt.figure(2, figsize= (10,6))
    plt.title("Mean capacity")
    max_y_reached = 0
    for property in env.all_property_list:
        plt.plot(property.capacity_episodes_mean, color=colors[property.id], label="Property "+str(property.id) if property.id != 0 else "Our property")
        tmp_max = max(property.capacity_episodes_mean)
        if tmp_max > max_y_reached:
            max_y_reached = tmp_max
    
    len_diff = len(x_days) - len(env.cnt_not_booked_customer_mean)
    if len_diff > 0:
        for _ in range(len_diff):
            env.cnt_not_booked_customer_mean.append(0)

    plt.bar(x_days, env.cnt_not_booked_customer_mean, alpha=0.4)
    plt.ylabel("Property mean capacity")
    plt.grid()

    y_ticks = get_y_ticks(max_y_reached)
    plt.yticks(y_ticks)
    plt.xticks(x_ticks)
    plt.xlabel('Days')

    if n_property < 10:
        plt.legend()

    plt.savefig(os.path.join(output_folderpath, 'Mean capacity.png'))

   
    ##################################
    # Chart 3 - Mean revenue
    ##################################

    plt.figure(3, figsize= (10,6))
    plt.title("Mean revenue")
    max_y_reached = 0
    for property in env.all_property_list:
        plt.plot(property.revenue_episodes_mean, color=colors[property.id], label="Property "+str(property.id) if property.id != 0 else "Our property")
        tmp_max = max(property.revenue_episodes_mean)
        if tmp_max > max_y_reached:
            max_y_reached = tmp_max
    plt.ylabel("Property mean revenue")
    plt.grid()

    y_ticks = get_y_ticks(max_y_reached)
    plt.yticks(y_ticks)
    plt.xticks(x_ticks)
    plt.xlabel('Days')

    if n_property < 10:
        plt.legend()

    plt.savefig(os.path.join(output_folderpath, 'Mean revenue.png'))

    ####################################
    # Chart 4 - Mean cumulative revenue
    ####################################

    plt.figure(4, figsize= (10,6))
    plt.title("Mean cumulative revenue")
    max_y_reached = 0
    for property in env.all_property_list:
        y_tmp = list(accumulate(property.revenue_episodes_mean))
        plt.plot(y_tmp, color=colors[property.id], label="Property "+str(property.id) if property.id != 0 else "Our property")
        tmp_max = max(y_tmp)
        if tmp_max > max_y_reached:
            max_y_reached = tmp_max
    plt.ylabel("Property mean cumulative revenue")
    plt.grid()

    y_ticks = get_y_ticks(max_y_reached)
    plt.yticks(y_ticks)
    plt.xticks(x_ticks)
    plt.xlabel('Days')

    if n_property < 10:
        plt.legend()

    plt.savefig(os.path.join(output_folderpath, 'Mean cumulative revenue.png'))

    ##################################
    # Chart 5 - Mean price
    ##################################

    plt.figure(5, figsize= (10,6))
    plt.title("Mean price")
    max_y_reached = 0
    for property in env.all_property_list:
        plt.plot(property.price_episodes_mean, color=colors[property.id], label="Property "+str(property.id) if property.id != 0 else "Our property")
        tmp_max = max(property.price_episodes_mean)
        if tmp_max > max_y_reached:
            max_y_reached = tmp_max
    plt.ylabel("Property mean price")
    plt.grid()

    y_ticks = get_y_ticks(max_y_reached)
    plt.yticks(y_ticks)
    plt.xticks(x_ticks)
    plt.xlabel('Days')

    if n_property < 10:
        plt.legend()

    plt.savefig(os.path.join(output_folderpath, 'Mean price.png'))

    ##################################
    # Chart 6 - Customer preferences
    ##################################

    plt.figure(6, figsize= (10,6))
    plt.subplot(1,2,1)
    plt.title("Customer bookings")
    total_customers_mean = sum(env.daily_arrivals_mean)
    total_customers_booking_mean = 0
    for property in env.all_property_list:
        total_customers_booking_mean += sum(property.how_many_bookings_mean)
    total_customers_not_booking_mean = total_customers_mean - total_customers_booking_mean
    perc_not_booked = (total_customers_not_booking_mean*100)/total_customers_mean
    perc_booked =  (total_customers_booking_mean*100)/total_customers_mean
    plt.pie([perc_booked, perc_not_booked], labels= ["Booked", "Not booked"], autopct='%1.1f%%', textprops=dict(color="w"))
    plt.legend()

    plt.subplot(1,2,2)
    plt.title("Customer preferences")
    perc_property_booking = [0 for _ in range(n_property)]
    for cnt_property, property in enumerate(env.all_property_list):
        perc_property_booking[cnt_property] = (sum(property.how_many_bookings_mean)*100)/total_customers_booking_mean
    plt.pie(perc_property_booking, colors=colors, labels=["Property "+str(index_property) if index_property != 0 else "Our property" for index_property in range(len(env.all_property_list))], autopct='%1.1f%%', textprops=dict(color="w"))

    if n_property < 10:
        plt.legend()

    plt.savefig(os.path.join(output_folderpath, 'Customers bookings and preferences.png'))


config_path = get_config_file()
with open(config_path) as config_file:
    config = toml.load(config_file)

datetime_str = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
output_folderpath = os.path.join(os.path.normpath(os.path.dirname(__file__) + os.sep + os.pardir + os.sep + os.pardir), "deleteme_results")
output_path = os.path.join(output_folderpath, datetime_str)

which_state = config['models']['state']
which_agent = config['models']['algorithm']
which_reward = config['models']['reward']
dynamic_price_competitors = config['simulation']['dynamic_price_competitors']
tot_episodes = config['simulation']['tot_episodes']
which_action = config['models']['action']

env = SimulationEnvironment(
    config=config, 
    dynamic_price_competitors=dynamic_price_competitors, 
    which_state=which_state, 
    which_action=which_action, 
    which_agent=which_agent, 
    which_reward=which_reward, 
    dynamic_our_property=True,
    draw_charts=True
)


model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=os.path.join(output_path, "tblogs")
)

model.learn(
    total_timesteps=config['simulation']['tot_episodes'],
    progress_bar=True
)


observation = env.reset()

n_steps = 10000
for episode in range(n_steps):

    action, _states = model.predict(observation)
    observation, reward, done, info = env.step(action=action)
    compute_cumulative_data_episodes(env)

    if done:
        observation = env.reset()

compute_mean_data_episodes(env, episode)
plot_data(env, output_path)