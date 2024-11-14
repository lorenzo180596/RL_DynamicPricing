"""

This module contain all the functionality to correctly store and save the data generated.

Classes
--------------------
In that file the following classes are defined:

1. Storage
   - define the paths for all the relevant data that need to be saved

"""

import os

class Storage():

    def __init__(self, 
                 config, 
                 training_flag,
                 validation_flag,
                 test_flag, 
                 dynamic_our_property, 
                 model_to_load_path):
        
        if training_flag:
            if model_to_load_path is None:
                suffix = "Training - part 1"
            else:
                path_splitted = model_to_load_path.split('/')
                sim_model = path_splitted[-2].split()[1]
                sim_part = int(path_splitted[-2].split()[6])+1
                suffix = f"Training - part {sim_part} - from Sim {sim_model}" 
               
        elif validation_flag:
            path_splitted = model_to_load_path.split('/')
            sim_model = path_splitted[-2].split()[1]
            suffix = f"Validation - from Sim {sim_model}"
        elif test_flag:
            if model_to_load_path is not(None):
                path_splitted = model_to_load_path.split('/')
                sim_model = path_splitted[-2].split()[1]
                if dynamic_our_property:
                    suffix = f"Test - from Sim {sim_model} - Dynamic ON"
                else:
                    suffix = f"Test - from Sim {sim_model} - Dynamic OFF"
            else:
                if dynamic_our_property:
                    suffix = f"Test - Dynamic ON"
                else:
                    suffix = f"Test - Dynamic OFF"

        self.current_path = os.path.abspath('')
        self.risultati_simulazione_path = os.path.join(self.current_path, config['storage']['folder_name'])
        if not os.path.exists(self.risultati_simulazione_path):
            os.makedirs(self.risultati_simulazione_path)

        subfolder_name = os.listdir(self.risultati_simulazione_path)
        if len(subfolder_name)>0:
            subfolder_name = subfolder_name[-1]
            num_folder = int(subfolder_name.split()[1])+1
        else:
            num_folder = 1
        if num_folder <10:
            num_folder = "00"+str(num_folder)
        elif num_folder <100:
            num_folder = "0"+str(num_folder)
        
        self.folder_path = os.path.join(self.risultati_simulazione_path, f'Sim {num_folder} - {suffix}')
        os.makedirs(self.folder_path)
        self.chart_path = os.path.join(self.folder_path, config['storage']['chart_folder_name'])
        os.makedirs(self.chart_path)
        self.file_path = os.path.join(self.folder_path, config['storage']['excel_name'] + f" - {num_folder}.xlsx")
        self.model_path = os.path.join(self.folder_path, config['storage']['network_name'])
        self.plot_training_path = os.path.join(self.folder_path, config['storage']['chart_name'])