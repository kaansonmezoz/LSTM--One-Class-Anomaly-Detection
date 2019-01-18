import read_from_file
import os.path
from creating_LSTM_model import creating_model
from file_operations import save_json
from dataset_preparation import get_datasets
from test_LSTM_model import test_ADL_validation_set
from test_LSTM_model import test_ADL_test_set
from test_LSTM_model import test_FALL_test_set
from keras.models import load_model
import numpy as np

### import combining_sensor_files sadece bir kere calistirilmali. MobiFall'da sensor dosyalari ayri bulunuyor. İlgili observationin sensor verilerini tek bir dosyada toplar. Bunu yaparken de resampling uygular.
### uyguladigi resampling fonksiyonu ve degerleri params icerisinde yer aliyor. Kodta ilgili yerin manuel degistirilmesi gerekiyor suanda. Daha sonra burası sadece bu parametre degistilerek yapilabilir hale getirilecek

### bu parametreler oncelikle dosyadan okunmaya calisilmasi gerekir aslında ... ya da model kaydedilmisse eger bu parametreler zaten dosya da 
### var demektir bu parametreler dosyaya surekli yaziliyor bunlarla ilgili  refactoring olmali. ayrica eger dosya halihazirda varsa yani model kayitliyse
### bizim bu parametreleri dosyadan okumamiz gerekiyorsa nasil modeli yukluyorsak


### defining hyper parameters such as fitting parameters, dataset frequency, window size and other parameters
### related to anomaly_detector

params = {
          'epochs': 50, 
          'batch_size': 128, 
          
          'window_size': 400,           ### sample_duration * resample_frequency seklinde bulunur
          'window_feature': 9,
          'sliding_window': 200, 
          
          'resample': {
                  'frequency': 100,
                   'method': 'resample_sensor_frequency_decimation' ### bu metod adını kullanabileceğimiz bir parametre kullanılmalı resampling icin birden farki yontem verebilir halde olalim modele
          },          
          
          'random_state': {         ### genel kullanılan random_state degerleri 24 ya da 42
              'test_set': 24,        
              'validation_set': 24
          },
          
          'cut_first_ADL': 2,                 ### bastan kac saniye kesilecegini belirtir. Kesilmeyecekse sifir verilmeli
          'cut_last_ADL': 2,                  ### sondan kac saniye kesilecegini belirtir. Kesilmeyecekse sifir verilmeli
          'cut_first_FALL': 2,
          'cut_last_FALL': 2,
          
          'normalization': False,
          'normalization_range': '(-1,1)'  ### Suggested options are '(-1,1)' and '(0,1)'
}

ADL_SET_PATH = "../resampled-data/ADL"
FALL_SET_PATH = "../resampled-data/FALL"
OUTPUT_DIRECTORY = "../models"
OUTPUT_MODEL_NAME = "model_26"

"""
        Burayi baska zaman tekrar yapmak gerekecek
        
RESAMPLED_DATA_DIRECTORY = '../resampled-data/' + params['resample']['frequency'] + '_' + params['resample']['method']

if not os.path.exists(RESAMPLED_DATA_DIRECTORY):

"""
    
beginning_index = params['cut_first_ADL'] * params['resample']['frequency']

if params['cut_last_ADL'] == 0:
    ending_index = None
else:
    ending_index = - (params['cut_last_ADL'] * params['resample']['frequency'])

train_set = read_from_file.get_samples(ADL_SET_PATH, OUTPUT_DIRECTORY + '/' + OUTPUT_MODEL_NAME + '/', beginning_index, ending_index, params['window_size'], params['sliding_window'])

train_set_ADL, validation_set_ADL, test_set_ADL = get_datasets(train_set, params['window_size'], params['random_state']['test_set'], params['random_state']['validation_set'])

ADL_set_size = train_set_ADL.shape[0] + validation_set_ADL.shape[0] + test_set_ADL.shape[0]

### Creating outputs for ADL class which is represented by 0
expected_output_train = np.zeros((train_set_ADL.shape[0], 1))
expected_output_test = np.zeros((test_set_ADL.shape[0], 1))  ### Output olarak herhangi bir time-series verisini tahmin etmesini istemiyoruz 0'a yakin sectim ki bu sayede en azından threshold degeri olarak daha alt degerler secmis olalim    
expected_output_validation = np.zeros((validation_set_ADL.shape[0], 1))

model_file_path = OUTPUT_DIRECTORY + '/' + OUTPUT_MODEL_NAME + '/' + OUTPUT_MODEL_NAME +'.h5'

if os.path.isfile(model_file_path):
    print('Model has been already created with this name. Model is loading.')
    
    model = load_model(model_file_path)
    model.summarize()
    
else:
    print('New model has been creating.')
    model = creating_model(params, OUTPUT_MODEL_NAME, train_set_ADL, expected_output_train)
    
min_value, max_value = test_ADL_validation_set(model, validation_set_ADL, OUTPUT_DIRECTORY + '/' + OUTPUT_MODEL_NAME, params)

save_json("../models/" + OUTPUT_MODEL_NAME, 'parameters', params)

actual_FALL_count = {'predicted_FALL': 0, 'predicted_ADL': 0} ### it stores count of the prediction types when FALL test set is given
actual_ADL_count = {'predicted_ADL': 0, 'predicted_FALL': 0} ### it stores count of the prediction types when ADL test set is given

test_ADL_test_set(model, test_set_ADL, min_value, max_value, actual_ADL_count, OUTPUT_DIRECTORY + '/' + OUTPUT_MODEL_NAME)

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

test_FALL_test_set(model, test_set_FALL, min_value, max_value, actual_FALL_count, OUTPUT_DIRECTORY + '/' + OUTPUT_MODEL_NAME)

from statistics import analyze_results

analyze_results({'ADL': actual_ADL_count, 'FALL': actual_FALL_count}, OUTPUT_DIRECTORY + '/' + OUTPUT_MODEL_NAME)