import read_from_file
import os.path
from creating_LSTM_model import creating_model
from creating_LSTM_model import save_params
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
### defining hyper parameters such as fitting parameters, dataset frequency, window size
params = {'epochs': 50, 
          'batch_size': 128, 
          'window_size': 300,   ### sample_duration * resample_frequency seklinde bulunur
          'sliding_window': 200, 
          'resample_frequency': 100,
          'resample_method': 'resample_sensor_frequency_decimation', ### bu metod adını kullanabileceğimiz bir parametre kullanılmalı resampling icin birden farki yontem verebilir halde olalim modele
          'random_state_for_test_set' : 24,
          'random_state_for_validation': 24,   ### genel kullanılan random_state degerleri 24 ya da 42
          'cut_first': 2,                 ### bastan kac saniye kesilecegini belirtir. Kesilmeyecekse sifir verilmeli
          'cut_last': 2                   ### sondan kac saniye kesilecegini belirtir. Kesilmeyecekse sifir verilmeli
}

ADL_SET_PATH = "../resampled-data/ADL"
FALL_SET_PATH = "../resampled-data/FALL"
OUTPUT_DIRECTORY = "../models"
OUTPUT_MODEL_NAME = "model_3"

### combining_sensor_files ile hem resampling olaylarini yapiyorduk bunu disaridan parametre alacak hale getirerek buradan calistirabilmemiz gerekir
### ve halihazirda o yontemle o frekansla kombine edilmisse dosyalar bunun kontrolü yapılmalı ki sürekli ayni tipte bir sey olusturulmasın

beginning_index = params['cut_first'] * params['resample_frequency']

if params['cut_last'] == 0:
    ending_index = None
else:
    ending_index = - (params['cut_last'] * params['resample_frequency'])

train_set = read_from_file.get_samples(ADL_SET_PATH, OUTPUT_DIRECTORY + '/' + OUTPUT_MODEL_NAME + '/', beginning_index, ending_index, params['window_size'], params['sliding_window'])

train_set_ADL, validation_set_ADL, test_set_ADL = get_datasets(train_set, params['window_size'], params['random_state_for_test_set'], params['random_state_for_validation'])

ADL_set_size = train_set_ADL.shape[0] + validation_set_ADL.shape[0] + test_set_ADL.shape[0]

### Creating outputs for ADL class which is represented by 0
expected_output_train = np.zeros((train_set_ADL.shape[0], 1))
expected_output_test = np.zeros((test_set_ADL.shape[0], 1))  ### Output olarak herhangi bir time-series verisini tahmin etmesini istemiyoruz 0'a yakin sectim ki bu sayede en azından threshold degeri olarak daha alt degerler secmis olalim    
expected_output_validation = np.zeros((validation_set_ADL.shape[0], 1))

model_file_path = OUTPUT_DIRECTORY + '/' + OUTPUT_MODEL_NAME + '/' + OUTPUT_MODEL_NAME +'.h5'

if os.path.isfile(model_file_path):
    ### load the model ...
    print('Model daha onceden olusturulmus bu ad ile. Model yükleniyor')
    model = load_model(model_file_path)
    
else:
    ### creating a new model
    print('Yeni model olusturuluyor')
    model = creating_model(params, OUTPUT_MODEL_NAME, train_set_ADL, expected_output_train)
    
### buralarda bir fonksiyon olarak yapılabilir aslında
    
min_value, max_value = test_ADL_validation_set(model, validation_set_ADL, OUTPUT_DIRECTORY + '/' + OUTPUT_MODEL_NAME, params)

save_params(OUTPUT_MODEL_NAME, params)

actual_FALL_count = {'predicted_FALL': 0, 'predicted_ADL': 0} ### it stores count of the prediction types when FALL test set is given
actual_ADL_count = {'predicted_ADL': 0, 'predicted_FALL': 0} ### it stores count of the prediction types when ADL test set is given

test_ADL_test_set(model, test_set_ADL, min_value, max_value, actual_ADL_count, OUTPUT_DIRECTORY + '/' + OUTPUT_MODEL_NAME)

### sadece son bes saniyeyi alarak yapmamiz gerekiyor ya da baska saniyeleri baz alarak suan alayini aliyoruz
### test_set icerisinde sadece fall'lar yok adl'lerde var bu acidan problemli bir durum bu kısma bakmamiz lazim yani
### adl bulunan eski dataframe'lerin icine ekliyor fall'lari
### burada bir hata var su sekilde adl'lerin basindan ve sonundan iki saniye kesmezsem eğer 0 ve None göndererek read_from_file.get_samples düzgün sonuç döndümüyor boş döndürüyor neden ?
### ayrıca hala dataframede su problemi cozmedik ayni memorydeki yere ekliyor belki bunun icin append kullanilabilir concat yerine bazi yerlerde dolayısıyla append vs concat'e daha iyi bakmak gerekebilir
test_set_FALL = read_from_file.get_samples(FALL_SET_PATH, OUTPUT_DIRECTORY + '/' + OUTPUT_MODEL_NAME + '/', beginning_index, ending_index, params['window_size'], params['sliding_window'])

### Shaping datasets as in windows
sample_amount = test_set_FALL.shape[0] // params['window_size']
test_set_FALL = test_set_FALL.values
test_set_FALL = test_set_FALL.reshape((sample_amount, params['window_size'] , 9))
test_set_FALL = test_set_FALL[ADL_set_size:] ## burada soyle bir problem var yeni bir dataframe yaratmak yerine halihazirdakine concat ediyor dolayısıyla her seferinde test_set_FALL'daki window'larda artıyor eger yeni bastan program calistirilmazsa. Her seferinde variable'lar temizlenmeli yani.

test_FALL_test_set(model, test_set_FALL, min_value, max_value, actual_FALL_count, OUTPUT_DIRECTORY + '/' + OUTPUT_MODEL_NAME)

# save_params(OUTPUT_MODEL_NAME, {'actual_output_fall': actual_FALL_count, 'actual_output_adl': actual_ADL_count}, 'predictions')

from statistics import analyze_results

analyze_results({'ADL': actual_ADL_count, 'FALL': actual_FALL_count}, OUTPUT_DIRECTORY + '/' + OUTPUT_MODEL_NAME)