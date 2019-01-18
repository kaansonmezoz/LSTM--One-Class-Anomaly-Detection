"""

Burada da sadece LSTM_modeli test edilmeli. Test set ile. Belki threshold değerlerini belirlemek için
olan işlemi validation set ile ilgili işlemlerde LSTM_model de olur. Ya da başka bir dosyada.

creating_anomaly_detector diye bir fonksyion olur ya da python dosyası orada LSTM yaratılır yani 
creating_LSTM_model ile LSTM_modeli oluşturulur. Sonra validation set ile threshold değerleri belirlenir.
test edilir ve statistics.py ile de modelin istatistikleri çıkartılır ne gibi işte confussion matrix
loss, accuracy gibi istatistikler çıkartılır. creating_anomaly_detector da dataset okunur ilgili dosyadan
ve burada validation,train,test setlerine ayristilir. creating_LSTM_model de create eden fonksiyona parametre
olarak mesela train set gider, bu fonksyiyon modeli return eder en son compile edilip fit edilmiş halini yani tam 
anlamiyla kullanima hazir halini. belki de train için de ayrı bir .py gerekebilir. creating_anomaly_detector mesela
command-line'dan calisabilir hale getirilir ilgili parametreleri alir resampling_frequency, epochs, batch_size, window_size, 
sliding_window gibi digerleri için gerekli olan ve elle değiştirebilir parametreler girilir. Ayrica şu da yapılabilir olmali bence modeli 
save ettiğimiz için nihayetinde load'da etme modülü olmalı bence. Önce bakar o model var mı yoksa gidip yaratabilir


"""

import numpy as np

def test_model(model, datasets):
    
    return 

### Validation tests are used for determining thresholds and other parameters
### This function returns the upper and lower threshold for our detector based on adl dataset
def test_ADL_validation_set(model, validation_set, OUTPUT_PATH, params):
    from visualization import plot_set_results

    predicted = model.predict(validation_set)

    ### thresholdları bulan bir fonksiyon olmali 
    max_value = np.max(predicted)
    min_value = np.min(predicted)

    params['lower_threshold'] = str(min_value)
    params['upper_threshold'] = str(max_value)

    plot_set_results(predicted, min_value, max_value, OUTPUT_PATH, 'prediction values of ADL validation set', 'upper left')
        
    return min_value, max_value

def test_ADL_test_set(model, test_set, lower_threshold, upper_threshold, actual_ADL_count, OUTPUT_PATH):
    from visualization import plot_set_results
    
    predicted = model.predict(test_set)
    size = len(predicted)
    
    for i in range(size):
        if predicted[i] > upper_threshold or predicted[i] < lower_threshold: ### exceeds thresholds so it should be an anomaly
            actual_ADL_count['predicted_FALL'] += 1
        else:
            actual_ADL_count['predicted_ADL'] += 1
    
    plot_set_results(predicted, lower_threshold, upper_threshold, OUTPUT_PATH, 'prediction values of ADL test set', 'best')
    
def test_FALL_test_set(model, test_set, lower_threshold, upper_threshold, actual_FALL_count, OUTPUT_PATH, fig_name = 'prediction_test_set_FALL'):
    from visualization import plot_set_results
    
    predicted = model.predict(test_set)
    size = len(predicted)
    
    for i in range(size):
        if predicted[i] > upper_threshold or predicted[i] < lower_threshold:
            actual_FALL_count['predicted_FALL'] += 1
        else:
            actual_FALL_count['predicted_ADL'] += 1

        plot_set_results(predicted, lower_threshold, upper_threshold, OUTPUT_PATH, fig_name, 'best')
    
    
    