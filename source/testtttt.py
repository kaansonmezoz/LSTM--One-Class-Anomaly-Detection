import read_from_file


ADL_SET_PATH = "../resampled-data/ADL"
FALL_SET_PATH = "../resampled-data/FALL"
OUTPUT_DIRECTORY = "../models"
OUTPUT_MODEL_NAME = "model_28"


params = {
    'cut_first_FALL': 3,
    'cut_last_FALL' : 3,
    'window_size': 300,
    
    'resample': {
        'frequency': 100        
    },
            
    'sliding_window': 600        
}

beginning_index = params['cut_first_FALL'] * params['resample']['frequency']

if params['cut_last_FALL'] == 0:
    ending_index = None
else:
    ending_index = - (params['cut_last_FALL'] * params['resample']['frequency'])

test_set_FALL = read_from_file.get_samples(FALL_SET_PATH, OUTPUT_DIRECTORY + '/' + OUTPUT_MODEL_NAME + '/', beginning_index, ending_index, params['window_size'], params['sliding_window'])

print("test_set_FALL.shape[0]: ", test_set_FALL.shape[0])

### Shaping datasets as in windows
sample_amount = test_set_FALL.shape[0] // params['window_size']
test_set_FALL = test_set_FALL.values
test_set_FALL = test_set_FALL.reshape((sample_amount, params['window_size'] , 9))
