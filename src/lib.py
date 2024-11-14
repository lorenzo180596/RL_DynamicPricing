import numpy as np
import os
import pandas as pd

from matplotlib import pyplot as plt
from operator import add 
from itertools import accumulate

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
    
def get_data_last_training_file(path_to_look_in):
    path_to_look_in = os.path.dirname(path_to_look_in)
    name = 'data_last_training.toml'
    data_last_training_path = os.path.join(path_to_look_in, name)
    return data_last_training_path

def get_memory_buffer(path_to_look_in):
    path_to_look_in = os.path.dirname(path_to_look_in)
    name = 'memory_buffer'
    memory_buffer_path = os.path.join(path_to_look_in, name)
    return memory_buffer_path

def save_data_to_excel(episode, env, config, state, n_states, storage, seed = None):
        index = [f"Property {index_property} - ep. {episode}"  if index_property != 0 else f"Our property - ep. {episode}" for index_property in range(len(env.all_property_list))]
        # Starting values
        starting_values_df = pd.DataFrame(data= [[property.rating_init, property.position, property.price_init, property.capacity_init, property.min_room_price] for property in env.all_property_list],
                                          index= index,
                                          columns= ["Rating start ", "Position", "Price start", "Capacity start", "Min room price"])
        # Day per day values
        everyday_price_df = pd.DataFrame(data= [[property.price_init] + property.price for property in env.all_property_list],
                                         index= index,
                                         columns= ["Starting price"]+["Price during day "+str(day) for day in range(env.tot_days)])

        everyday_capacity_df = pd.DataFrame(data= [[property.capacity_init] + property.capacity for property in env.all_property_list]+[["-"]+env.cnt_not_booked_customer],
                                            index= index + [f"No bookings - ep. {episode}"],
                                            columns= ["Starting capacity"]+["Capacity after day "+str(day) for day in range(env.tot_days)])
        # Customer coming every day
        everyday_customer_df = pd.DataFrame(data= [env.demand.daily_arrivals, [len(customers_coming_in_a_day) for customers_coming_in_a_day in env.pending_customers]],
                                            index= [f"New customers - ep. {episode}", f"Total customers - ep. {episode}"],
                                            columns= ["Day "+str(day) for day in range(env.tot_days)])
        # Revenue
        revenue_df = pd.DataFrame(data= [property.revenue+[sum(property.revenue)] for property in env.all_property_list],
                                  index= index,
                                  columns= ["Revenue after day "+str(day) for day in range(env.tot_days)]+["Cumulative revenue"])
        
        # Losses
        losses_df = pd.DataFrame(data= [property.losses+[sum(property.losses)]  for property in env.all_property_list],
                                  index= index,
                                  columns= ["Losses after day "+str(day) for day in range(env.tot_days)]+["Cumulative losses"])
        
        # Environment state
        if env.env_settings.discrete:
            state_df = pd.DataFrame(data= [[state[which_day][which_state] for which_day in range(env.tot_days+1)] for which_state in range(n_states)],
                                    index= [label + f" - ep. {episode}" for label in env.state_labels],
                                    columns= ["Start of day "+str(day) if day < env.tot_days else "End of day "+str(day-1) for day in range(env.tot_days+1)])
        
        #Seed used
        if config['simulation']['use_custom_seed']:
            seed_df = pd.DataFrame(data= seed,
                                index= [f"Episode {episode}"],
                                columns= ["Seed used"])

        # Print data to Excel:
        if os.path.isfile(storage.file_path):
            with pd.ExcelWriter(storage.file_path, mode="a", if_sheet_exists="overlay") as writer:
                starting_values_df.to_excel(writer, sheet_name = 'Starting_values', startrow=writer.sheets['Starting_values'].max_row)
                everyday_price_df.to_excel(writer, sheet_name = 'Everyday_price', startrow=writer.sheets['Everyday_price'].max_row)
                everyday_capacity_df.to_excel(writer, sheet_name = 'Everyday_capacity', startrow=writer.sheets['Everyday_capacity'].max_row)
                everyday_customer_df.to_excel(writer, sheet_name = 'Everyday_customers', startrow=writer.sheets['Everyday_customers'].max_row)
                revenue_df.to_excel(writer, sheet_name = 'Revenues', startrow=writer.sheets['Revenues'].max_row)
                losses_df.to_excel(writer, sheet_name = 'Losses', startrow=writer.sheets['Losses'].max_row)
                if env.env_settings.discrete:
                    state_df.to_excel(writer, sheet_name = 'State', startrow=writer.sheets['State'].max_row)
                if config['simulation']['use_custom_seed']:
                    seed_df.to_excel(writer, sheet_name = 'Seed', startrow=writer.sheets['Seed'].max_row)
                
        else:
            with pd.ExcelWriter(storage.file_path, mode="w") as writer:
                starting_values_df.to_excel(writer, sheet_name= 'Starting_values')
                everyday_price_df.to_excel(writer, sheet_name= 'Everyday_price')
                everyday_capacity_df.to_excel(writer, sheet_name= 'Everyday_capacity')
                everyday_customer_df.to_excel(writer, sheet_name= 'Everyday_customers')
                revenue_df.to_excel(writer, sheet_name= 'Revenues')
                losses_df.to_excel(writer, sheet_name= 'Losses')
                if env.env_settings.discrete:
                    state_df.to_excel(writer, sheet_name= 'State')
                if config['simulation']['use_custom_seed']:
                    seed_df.to_excel(writer, sheet_name= 'Seed')
                empty_df = pd.DataFrame(data= [""],
                                  index= [""],
                                  columns= [""])
                empty_df.to_excel(writer, sheet_name= 'Time')

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

def get_y_ticks(max_y_reached):
    resolution = (max_y_reached//50)*5 + 5
    max_y_reached = max_y_reached + (resolution - (max_y_reached % resolution))
    y_ticks = [cnt_ticks*resolution for cnt_ticks in range(int(max_y_reached/resolution)+1)]
    return y_ticks

def get_x_ticks(env):
    resolution = (env.tot_days//50)*2 + 2
    tot_days_ticks = env.tot_days + (resolution - (env.tot_days % resolution))
    x_ticks = [cnt_ticks*resolution for cnt_ticks in range(int(tot_days_ticks/resolution))]  
    return x_ticks

def plot_data(env, storage):

    x_ticks = get_x_ticks(env)

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

    plt.savefig(os.path.join(storage.chart_path, 'Mean daily arrivals.png'))


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

    plt.savefig(os.path.join(storage.chart_path, 'Mean capacity.png'))

   
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

    plt.savefig(os.path.join(storage.chart_path, 'Mean revenue.png'))

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

    plt.savefig(os.path.join(storage.chart_path, 'Mean cumulative revenue.png'))

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

    plt.savefig(os.path.join(storage.chart_path, 'Mean price.png'))

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

    plt.savefig(os.path.join(storage.chart_path, 'Customers bookings and preferences.png'))

def sim_episode(epsilon_array, episode, env, agent_active = True):
    # initialize the state
    day = 0
    state = env.reset()
    state = [[] if cnt>0 else state for cnt in range(env.tot_days+1)]
    done = False
    rewards = 0
    epsilon = epsilon_array[episode]
    while not(done):
        if agent_active:
            action = env.agent.act(state[day], epsilon)
        else:
            action = None
        state[day+1], reward, done, info = env.step(action= action)
        if agent_active:
            env.agent.step(state[day], action, reward, state[day+1], done)
        rewards += reward
        day += 1
    
    return state, rewards, epsilon

def validate_NN(env, env_validation_no_dynamic, episode, rev_dyn, rev_no_dyn, winner_env, rev_diff_percentage):
    rev_dyn[episode] = sum(env.our_property.revenue)
    rev_no_dyn[episode] = sum(env_validation_no_dynamic.our_property.revenue)
    winner_env[episode] = 0 if rev_no_dyn[episode] >= rev_dyn[episode] else 1
    rev_diff_percentage[episode] = ((rev_dyn[episode] - rev_no_dyn[episode])/rev_no_dyn[episode])*100 if rev_no_dyn[episode] != 0 else 1000

def plot_validation_chart(winner_env, rev_diff_percentage, storage):
    
    fig, (ax1, ax2) = plt.subplots(2,1)
    how_many_wins_dyn = winner_env.count(1)
    how_many_wins_no_dyn = winner_env.count(0)
    ax1.pie([how_many_wins_dyn, how_many_wins_no_dyn], labels= ["With NN", "Without NN"], autopct='%1.1f%%', textprops=dict(color="w"))
    ax2.plot(rev_diff_percentage)
    rev_diff_percentage_filtered = [i for i in rev_diff_percentage if i != 1000]
    ax2.plot([0,len(rev_diff_percentage)-1],[np.mean(rev_diff_percentage_filtered),np.mean(rev_diff_percentage_filtered)],'r--')

    ax1.title.set_text("Winning environment")
    ax1.legend(bbox_to_anchor = (2, 0.5), loc='center right')
    ax2.title.set_text("Percentage increment with NN")
    ax2.grid()

    fig.savefig(os.path.join(storage.folder_path, 'NN policy validation.png'))    

