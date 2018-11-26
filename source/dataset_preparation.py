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

def get_expected_output_ADL(train_amount, validation_amount, test_amount):
    return 