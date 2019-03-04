from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Reshape
from keras.layers import Input
from keras.layers import InputLayer
from keras.layers import Flatten
from visualization import plot_loss
from file_operations import save_model
    
def creating_model(params, OUTPUT_MODEL_NAME, train_set_ADL, expected_output_train):
    ### Creating LSTM architecture 
    model = Sequential()
    
    ### Adding 1st LSTM Layer input_shape = (batch_size, timesteps, input_dimensions) 
    model.add(LSTM(units = 200, return_sequences = True, input_shape = ((params['window_size'],params['window_feature']))))

    ### Adding 2nd LSTM Layer
    model.add(LSTM(units = 200, return_sequences = False, dropout = params['lstm-dropout-2']))
    
    ### Adding the first Feed-Forward layer with 200 units
    model.add(Dense(units = 200))

    ### Adding the last Feed-Forward layer with 2 units ve bu final decision olan Y'yi uretir ki bunu da threshold
    ### degeri belirlerken kullanacagiz
    model.add(Dense(units = 2))

    ### Adding the output layer
    model.add(Dense(units = 1, ))
    model.summary()

    ### Configuring optimizer  burada bir kontrol konulmali ileride ki optimizer degisirse ilgili optimizer gelsin
    ### optimizer = params['optimizer'] bunu kullanip ilgili kontrol yapilip sonra da optimizer alinmali
    
    adam = keras.optimizers.Adam(lr = params['learning-rate'])
    
    ### Compiling LSTM    
    model.compile(loss = params['loss_function'],  optimizer = adam)

    ### Fitting LSTM to the training set
    history = model.fit(train_set_ADL, expected_output_train, epochs = params['epochs'], batch_size = params['batch_size'])

    save_model('../models/' + OUTPUT_MODEL_NAME, OUTPUT_MODEL_NAME, model)
    plot_loss(history, "../models", OUTPUT_MODEL_NAME, params)
    
    return model
