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
    import pandas as pd
    
    sensor_range_max = {'acc-x': [], 'acc-y': [], 'acc-z': [], 'ori-x': [], 'ori-y': [], 'ori-z': [], 'gyro-x': [], 'gyro-y': [], 'gyro-z': []}
    sensor_range_min = {'acc-x': [], 'acc-y': [], 'acc-z': [], 'ori-x': [], 'ori-y': [], 'ori-z': [], 'gyro-x': [], 'gyro-y': [], 'gyro-z': []}
    
    
    for i in range(len(dataset['acc-x'])):
    
        sensor_range_max['acc-x'].append(dataset.max()['acc-x'])  ### Mobifall_2.0 datasetinde olcumde kullanilan sensorlerin olcebilecegi araliklar  baz alinarak bu degerler belirlendi
        sensor_range_max['acc-y'].append(dataset.max()['acc-y'])
        sensor_range_max['acc-z'].append(dataset.max()['acc-z'])

        sensor_range_min['acc-x'].append(dataset.min()['acc-x'])
        sensor_range_min['acc-y'].append(dataset.min()['acc-y'])
        sensor_range_min['acc-z'].append(dataset.min()['acc-z'])

        sensor_range_max['ori-x'].append(dataset.max()['ori-x'])
        sensor_range_max['ori-y'].append(dataset.max()['ori-y'])
        sensor_range_max['ori-z'].append(dataset.max()['ori-z'])

        sensor_range_min['ori-x'].append(dataset.min()['ori-x'])
        sensor_range_min['ori-y'].append(dataset.min()['ori-y'])
        sensor_range_min['ori-z'].append(dataset.min()['ori-z'])

        sensor_range_max['gyro-x'].append(dataset.max()['gyro-x'])
        sensor_range_max['gyro-y'].append(dataset.max()['gyro-y'])
        sensor_range_max['gyro-z'].append(dataset.max()['gyro-z'])

        sensor_range_min['gyro-x'].append(dataset.min()['gyro-x'])
        sensor_range_min['gyro-y'].append(dataset.min()['gyro-y'])
        sensor_range_min['gyro-z'].append(dataset.min()['gyro-z'])

    df_max = pd.DataFrame.from_dict(sensor_range_max)
    df_min = pd.DataFrame.from_dict(sensor_range_min)
    
    dataset = (2 * (dataset - df_min) / (df_max - df_min)) - 1

    print(dataset)

    return dataset