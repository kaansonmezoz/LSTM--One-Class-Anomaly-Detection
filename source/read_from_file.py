import pandas as pd
import os

## COMBINED_DATA_PATH = "../resampled-data/train"  ### mesela bu disaridan bir parametre olarak gelmeli kod icerisine

def get_data_from_files(file_paths):
    sensor_datas = []
    size = len(file_paths)
    
    for i in range(size):
        ### sensor_data = pd.read_csv(file_paths[i], sep = '\t', header = None, skiprows = 1)
        sensor_data = pd.read_csv(file_paths[i], sep = '\t', header = None, names = ['acc-x', 'acc-y', 'acc-z', 'gyro-x', 'gyro-y', 'gyro-z', 'ori-x', 'ori-y', 'ori-z'], skiprows = 1)        
        sensor_datas.append({'sensor_data': sensor_data, 'file_name': file_paths[i]})
    
    return sensor_datas     ## bunlar data frame tabii list icindeki dataframe'ler var onemli 

def get_file_paths(folder_path, file_paths = []):
    for file in os.listdir(folder_path):
        file_path = folder_path + "/" + file
        
        if os.path.isdir(file_path):
            get_file_paths(file_path, file_paths)
        else:      
            file_paths.append(file_path)
    
    return file_paths

def create_window_dataframe(window_data):
    dataframe = {
                'acc-x' : window_data['acc-x'],
                'acc-y' : window_data['acc-y'],
                'acc-z' : window_data['acc-z'],
                
                'gyro-x' : window_data['gyro-x'],
                'gyro-y' : window_data['gyro-y'],
                'gyro-z' : window_data['gyro-z'],
                
                'ori-x' : window_data['ori-x'],
                'ori-y' : window_data['ori-y'],
                'ori-z' : window_data['ori-z']
    }
    
    return pd.DataFrame(data = dataframe)


### window_size = 300 oluyor bu durumda 3 saniye demek oluyor
### sliding_window = 200 bu da oluyor referans alınan noktadan sonra iki saniye kaydirilacak window
def get_windows_from_data(sensor_data, window_size, sliding_window, sample_info):
    
    window_dataframes = pd.DataFrame(columns = get_column_names())
    
    data_size = len(sensor_data)
    
    for i in range(0, data_size - window_size, sliding_window):
        window_dataframe = create_window_dataframe(sensor_data[i : i + window_size])
        ###print("\n---------  window_dataframe  ----------\n", window_dataframe)        
        window_dataframes = pd.concat([window_dataframes, window_dataframe], ignore_index = True)
        ###print("\n--------  window_dataframes ------------\n", window_dataframes)
        sample_info['sample_count'] += 1
        
    #print('Data Sequence Length', window_dataframes.shape[0])
    
    return window_dataframes

def get_sample_windows(observation_datas, window_size, sliding_window, sample_infos, beginning_index, ending_index):
    number_of_observations = len(observation_datas)

    samples = pd.DataFrame(columns = get_column_names())
    
    for i in range(number_of_observations):
        observation_data = observation_datas[i]['sensor_data']
        file_name = observation_datas[i]['file_name']
        
        observation_data = observation_data.iloc[beginning_index : ending_index, :]
        
        if 'ADL' in file_name:
            file_type = 'ADL'
        else:
            file_type = 'FALL'
        file_index = file_name.split('/')[-1].split('_')[0]
        sample_info = sample_infos[file_type]['codes'][file_index]
        
        sample_windows = get_windows_from_data(observation_data, window_size, sliding_window, sample_info)
        samples = pd.concat([samples, sample_windows], ignore_index = True)        
    
    #sample_count = len(samples) // window_size
    
    # print('Sample_count: ', sample_count)
    print('Window count: ', samples.shape[0] // window_size)
    
    return samples

def get_column_names():
    columns = ['acc-x', 'acc-y', 'acc-z', 'gyro-x', 'gyro-y', 'gyro-z', 'ori-x', 'ori-y', 'ori-z']
    
    return columns

def get_samples(train_folder_path, OUTPUT_PATH, beginning_index, ending_index, window_size = 300, sliding_window = 200):
    import json 
    import os
    
    sample_info = {'ADL': get_ADL_sample_info(), 'FALL': get_FALL_sample_info()}
        
    file_paths = get_file_paths(train_folder_path)        
    sensor_datas = get_data_from_files(file_paths)    
    samples = get_sample_windows(sensor_datas, window_size, sliding_window, sample_info, beginning_index, ending_index)
    
    del file_paths[ : ]

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    
    with open(OUTPUT_PATH + 'sample_info.json', 'w') as file:
        file.write(json.dumps(sample_info, indent = 4, sort_keys = True))
    
    return samples

def get_FALL_sample_info():
    fall_samples = {'codes': {
    'FOL':{
     'activity': 'Forward-lying',
     'description': 'Fall Forward from standing, use of hands to dampen fall',         
     'sample_count': 0,
     'activity_duration': '10s'
    },
    'FKL':{
     'activity': 'Front-knees-lying',
     'description': 'Fall forward from standing, first impact on knees',         
     'sample_count': 0,
     'activity_duration': '10s'
    },
    'BSC':{
     'activity': 'Back-sitting-chair',
     'description': 'Fall backward while trying to sit on a chair',
     'sample_count': 0,
     'activity_duration': '10s'
    },
    'SDL':{
     'activity': 'Sideward-lying',
     'description': 'Fall sidewards from standing, bending legs',
     'sample_count': 0,
     'activity_duration': '10s'   
    }
    }}
    
    return fall_samples

def get_ADL_sample_info():
    adl_samples = { 'codes': {
            'STD':{
                 'activity': 'Standing',
                 'description': 'Standing with subtle movements',
                 'sample_count': 0,
                 'activity_duration': '5m'  
            },
            'WAL':{            
                 'activity': 'Walking',
                 'description': 'Normal walking',
                 'sample_count': 0,
                 'activity_duration': '5m'  
            },
            'JOG':{
                 'activity': 'Jogging',
                 'description': 'Jogging',
                 'sample_count': 0,
                 'activity_duration': '30s'  
            },                    
            'JUM':{
                 'activity': 'Jumping',     
                 'description': 'Continuous jumping',
                 'sample_count': 0,
                 'activity_duration': '30s'  
            },
            'STU':{
                 'activity': 'Stairs up',
                 'description': 'Stairs up (10 stairs)',
                 'sample_count': 0,
                 'activity_duration': '10s'  
            },
            'STN':{
                 'activity': 'Stairs down',
                 'description': 'Stairs down (10 stairs)',
                 'sample_count': 0,
                 'activity_duration': '10s'  
            },                    
            'SCH':{
                 'activity': 'Sit chair',
                 'description': 'Sitting on a chair',
                 'sample_count': 0,
                 'activity_duration': '6s'  
            },                  
            'CSI':{
                 'activity': 'Car-step in',
                 'description': 'Step in a car',
                 'sample_count': 0,
                 'activity_duration': '6s'  
            },                  
            'CSO':{
                 'activity': 'Car-step out',
                 'description': 'Step out a car',
                 'sample_count': 0,
                 'activity_duration': '10s'  
            },                  
    }}
            
    return adl_samples