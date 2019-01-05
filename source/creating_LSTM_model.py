import json   ## Bu kaldirilacak daha sonra save_model silinecek cunku
import os     ## Bu kaldirilacak daha sonra save_model silinecek cunku 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Reshape
from keras.layers import Input
from keras.layers import InputLayer
from keras.layers import Flatten
from visualization import plot_loss

def save_params(model_name, fitting_params, file_name = 'parameters'):
    folder_path = "../models/" + model_name + '/'
    
    with open(folder_path + file_name + '.json', 'w') as file:
        file.write(json.dumps(fitting_params, indent = 4, sort_keys = True))

def save_model(model, model_name, fitting_params):
    folder_path = "../models/" + model_name + '/'
    json_string = model.to_json()  ### getting json of model architecture as in string
    
    model_json = json.loads(json_string)  ### converting json string to json (dict)
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    with open(folder_path + model_name + '.json', 'w') as file:
        json.dump(model_json, file, indent = 4, sort_keys = True)
    
    save_params(model_name, fitting_params)
    
    model.save(folder_path + model_name + '.h5')
    
def creating_model(params, OUTPUT_MODEL_NAME, train_set_ADL, expected_output_train):
    ### Creating LSTM architecture 
    model = Sequential()
    
    #model.add(Input(shape=(params['window_size']*9)))
    model.add(InputLayer((params['window_size'], 9)))
    
    # target_shape=(timestep, feature)  timestep * feature = window_size * 9 olmali
    model.add(Reshape(target_shape=(params['timestep_size'], params['timestep_feature'])))
    
    ### Adding 1st LSTM Layer input_shape = (batch_size, timesteps, input_dimensions) 
    model.add(LSTM(units = 200, return_sequences = True, input_shape = ((params['timestep_size'],params['timestep_feature']))))

    ### Adding 2nd LSTM Layer
    model.add(LSTM(units = 200, return_sequences = False))
    
    ### Adding the first Feed-Forward layer with 200 units
    model.add(Dense(units = 200))

    ### Adding the last Feed-Forward layer with 2 units ve bu final decision olan Y'yi uretir ki bunu da threshold
    ### degeri belirlerken kullanacagiz
    model.add(Dense(units = 2))

    ### Adding the output layer
    model.add(Dense(units = 1, ))
    model.summary()

    ### Compiling LSTM
    model.compile(loss = 'mean_squared_error', optimizer = 'adam')

    ### Fitting LSTM to the training set
    history = model.fit(train_set_ADL, expected_output_train, epochs = params['epochs'], batch_size = params['batch_size'])

    save_model(model, OUTPUT_MODEL_NAME, params)
    plot_loss(history, "../models", OUTPUT_MODEL_NAME, params)
    
    return model
