from sklearn.model_selection import train_test_split

def split_datasets(train_set, random_state_test, random_state_validation):
    ### Splitting dataset into train set, test set, validation set (60%, 20%, 20%)
    train_set_ADL, test_set_ADL = train_test_split(train_set, test_size = 0.20, random_state = random_state_test)  ### 80% train, 20% test
    train_set_ADL, validation_set_ADL = train_test_split(train_set_ADL, test_size = 0.25, random_state = random_state_validation) ### 25% 0f 80% equals the 20% of the total
    
    return train_set_ADL, validation_set_ADL, test_set_ADL

def reshape_data_frame_for_LSTM(data_set, window_size):
    sample_amount = data_set.shape[0] // window_size
    data_set = data_set.values
    data_set = data_set.reshape((sample_amount, window_size , 9))
    
    return data_set
    
def get_datasets(train_set_ADL, window_size, random_state_test, random_state_validation):
    train_set_ADL = reshape_data_frame_for_LSTM(train_set_ADL, window_size)
    
    return split_datasets(train_set_ADL, random_state_test, random_state_validation)

"""
    TODO: Normalizasyondaki min max'ı datasete gore belirleyelim yoksa problem yani bu boyle olmaz degerler fazla
    kucuk cikiyor isimizi zorlastiriyorlar fazlasiyla. FALL'lar icin baska secenekler dusunmek lazim bu arada ne tarz bir sey
    dusunebiliriz bilmiyorum ama bir halletmek lazim mantiken dimi ???
    
    TODO:  Daha iyi bir sekilde kesilmesi gerekiyor bu durumlarin ki daha rahat bir sekilde isimizi gorelim.
    Bu nasil yapilir gene kafalarda soru isareti var kabul ediyorum bu durumu. Ayrica normalizasyon range'ine gore 
    de bir kere denemek lazim yani bunu olabildigince formüle edelim de bir güzel rahatlayalim mesela 0,1 arasinda bir
    normalizasyon denemek daha iyi olabilir belki de.
    
    TODO: Bir de hem -1,1 hem de 0,1 arasinda normalizasyon yapayarak deneyelim ki ayni zamanda boyle bir islem icin
    hangi normalizasyon degerleri daha onemli olabilir bununla da ilgili bir sonuc elde edebilmis oluruz
    
    TODO: Beyaz tahta gibi bir seye oynayabilecegim parametreler ve bunlara vermek istedigim degerler ile ilgili seyler
    yazmak lazim. Ki daha rahat belli bir standartımız olabilsin.
"""

def normalization(dataset, downer_bound, upper_bound): ### acc-x acc-y acc-z ori-x ori-y ori-z gyro-x gyro-y gyro-z    
    g = 9.81

    ### based on the sensors used to obtain samples (reference: Analysis of Public Datasets for Wearable Fall Detection Systems)
    sensor_max_value = {
        'acc-x': 1.999 * g, 
        'acc-y': 1.999 * g, 
        'acc-z': 1.999 * g, 
        'ori-x': 360, 
        'ori-y': 360, 
        'ori-z': 360, 
        'gyro-x': 10.007805, ###573.42, Normalde max deger bu sensorun olcebilecegi ama textlerdeki max deger 10.007805 olmus bu kullanildiki elde edilen olcek degeri cok kucuk olmasin
        'gyro-y': 10.007805, ###573.42, 
        'gyro-z': 10.007805, ###573.42
    }  
    
    sensor_min_value = {
        'acc-x': -1.9951 * g,
        'acc-y': -1.9951 * g, 
        'acc-z': -1.9951 * g, 
        'ori-x': -179.9995,     
        'ori-y': -179.9995,
        'ori-z': -179.9995, 
        'gyro-x': -10.007500,  #-573.44, normalde min deger bu sensorun olcebilecegi ama textlerde kullanilan min deger -10.007500 olmus diger turlu -573.44'u kullaninca fazla kucuk oluyor
        'gyro-y': -10.007500,  #-573.44, 
        'gyro-z': -10.007500,  #-573.44 
    }
    
    for col_name in dataset.columns:
        max_value = sensor_max_value[col_name] ### bunun yerine daha fix bir scale edilebilmesi adina sensorlerin araliklarini koyacaktik hatirlarsan
        min_value = sensor_min_value[col_name]
        
        ### Normalizing dataset into (-1, 1) 
        dataset[col_name] = ((upper_bound - downer_bound) * (dataset[col_name] - min_value) / (max_value - min_value)) + downer_bound    

    return dataset