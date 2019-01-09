from sklearn.model_selection import train_test_split

def split_datasets(train_set, random_state_test, random_state_validation):
    ### Splitting dataset into train set, test set, validation set
    train_set_ADL, test_set_ADL = train_test_split(train_set, test_size = 0.20, random_state = random_state_test)  ### önce %80 train, %20 test
    train_set_ADL, validation_set_ADL = train_test_split(train_set_ADL, test_size = 0.25, random_state = random_state_validation) ### %80'in %25'i totaldeki verinin %20sine denk geliyor
    
    return train_set_ADL, validation_set_ADL, test_set_ADL

def reshape_data_frame_for_LSTM(data_set, window_size):
    sample_amount = data_set.shape[0] // window_size
    data_set = data_set.values
    data_set = data_set.reshape((sample_amount, window_size , 9))
    
    return data_set
    
def get_datasets(train_set_ADL, window_size, random_state_test, random_state_validation):
    train_set_ADL = reshape_data_frame_for_LSTM(train_set_ADL, window_size)
    
    return split_datasets(train_set_ADL, random_state_test, random_state_validation)

def normalization(dataset): ### acc-x acc-y acc-z ori-x ori-y ori-z gyro-x gyro-y gyro-z    
    g = 9.81

    ### based on the sensors used to obtain samples (reference: Analysis of Public Datasets for Wearable Fall Detection Systems)
    sensor_max_value = {
        'acc-x': 1.999 * g, 
        'acc-y': 1.999 * g, 
        'acc-z': 1.999 * g, 
        'ori-x': 360, 
        'ori-y': 360, 
        'ori-z': 360, 
        'gyro-x': 573.42, 
        'gyro-y': 573.42, 
        'gyro-z': 573.42
    }  
    
    sensor_min_value = {
        'acc-x': -1.9951 * g,
        'acc-y': -1.9951 * g, 
        'acc-z': -1.9951 * g, 
        'ori-x': -179.9995,     
        'ori-y': -179.9995,
        'ori-z': -179.9995, 
        'gyro-x': -573.44, 
        'gyro-y': -573.44, 
        'gyro-z': -573.44 
    }
    
    for col_name in dataset.columns:
        max_value = sensor_max_value[col_name] ### bunun yerine daha fix bir scale edilebilmesi adina sensorlerin araliklarini koyacaktik hatirlarsan
        min_value = sensor_min_value[col_name]
        
        ### Normalizing dataset into (-1, 1) 
        dataset[col_name] = (2 * (dataset[col_name] - min_value) / (max_value - min_value)) - 1    

    return dataset
