import os
import pandas as pd

FOLDER_PATH = "../data/MobiFall_Dataset_v2.0"
COMBINED_DATA_PATH_ADL = "../resampled-data/ADL"
COMBINED_DATA_PATH_FALL = "../resampled-data/FALL"

def get_ADL_files(folder_path, file_paths = []):
    for file in os.listdir(folder_path):
        file_path = folder_path + "/" + file
        if os.path.isdir(file_path) and "FALL" not in file_path: ### burayı daha sonra tekrardan elden gecirmek             
            file_paths = get_ADL_files(file_path, file_paths)    ### gerekecek bu sayede fall'ların da pathlerini bir yerde tutarız hem
        elif "FALL" not in file_path:       ### adl ve fall'i iki farkli listede tutariz o sekilde yeniden bir seyler yapabiliriz
            file_paths.append(file_path) ### ya da genel olarak bilmiyorum acc'leri alıp onlari combine edip bir yere koymak daha kolay olabilir yani ayri ayri fall mi bu adl mi diye bakmaktansa hani direkt acc gecenleri al onlari diger 2 sensor ile de birlestir bir dosyaya kaydet ve daha sonra adl mi lazim bu pathleri dondur, fall mi lazim bu pathleri dondur seklinde bir yol izlenebilir
    return file_paths

def get_FALL_files(folder_path, file_paths = []):
    for file in os.listdir(folder_path):
        file_path = folder_path + '/' + file        
        if os.path.isdir(file_path) and "ADL" not in file_path:
            file_paths = get_FALL_files(file_path, file_paths)
        elif "ADL" not in file_path:
            file_paths.append(file_path)
    
    return file_paths

def get_files(folder_path, file_paths = {'ADL': [], 'FALL': []}):
    
    for file in os.listdir(folder_path):
        file_path = folder_path + '/' + file
        
        if os.path.isdir(file_path):
            get_files(file_path, file_paths)        
        elif "ADL" in file_path:
            file_paths['ADL'].append(file_path)        
        else:
            file_paths['FALL'].append(file_path)
    
    return file_paths

def get_acc_files(file_paths):
    acc_files = []
    size = len(file_paths)
    
    for i in range(size):
        if "_acc_" in file_paths[i]:
            ### print(file_paths[i])
            acc_files.append(file_paths[i])
    return acc_files

def acc_path_to_ori(acc_file):
    ori_file = acc_file.replace("_acc_", "_ori_")
    return ori_file

def acc_path_to_gyro(acc_file):
    gyro_file = acc_file.replace("_acc_", "_gyro_")
    return gyro_file

def combine_sensor_files(acc_files, COMBINED_DATA_PATH):
    min_acc_freq = 80
    size = len(acc_files)
    count = 0
    
    for i in range(size):
        gyro_file, ori_file = convert_acc_file_path(acc_files[i])
        ### !!!! ori_datalar textte timestamp-z-x-y seklinde yer aliyor !!! Diger dosyalardakilerinde de degistirmek lazim !!!
        ### read_csv'deki siralamayi timestamp z x y seklinde yapmak lazim
        ### ama benim resample edip kaydettigim dosyalarda x-y-z seklinde gidiyor oluyor
        acc_data = pd.read_csv(acc_files[i], header= None, index_col = 0, names = ["timestamp", "x", "y", "z"], skiprows = 16)
        gyro_data = pd.read_csv(gyro_file, header= None, index_col = 0, names = ["timestamp", "x", "y", "z"], skiprows = 16)
        ori_data = pd.read_csv(ori_file, header= None, index_col = 0, names = ["timestamp", "z", "x", "y"], skiprows = 16)        
        
        event_periods = get_event_periods(acc_files[i], gyro_file, ori_file)

        acc_freq = calculate_frequency(event_periods[0], acc_data.shape[0])

        if acc_freq >= min_acc_freq:
            print("Filename : ", acc_files[i])
            count = count + 1
            
            dataframe = resample_sensor_frequency_decimation(acc_data, gyro_data, ori_data,event_periods[0])
            ### BUNDAN sonraki kısım daha efektif olabilirdi sanki
            file_path = acc_files[i].replace(FOLDER_PATH, COMBINED_DATA_PATH)
            file_name = file_path.split("/")[-1]            

            folder_path = file_path.replace("/" + file_name, "")
            file_name = file_name.replace("_acc_", "_")
            
            if not os.path.exists(folder_path):
                os.makedirs(folder_path) ### subdirectory'leri de create ediyor
            
            file_path = folder_path + "/" + file_name
            dataframe.to_csv(file_path, sep = '\t', index = False)
            

    print(">= 80Hz olan dosya sayisi: ", count)
    
    return

def combine_sensor_files_2(file_paths):
    min_acc_freq = 80
    count = 0
    
    file_paths = file_paths['ADL'] + file_paths['FALL']
    
    acc_files = get_acc_files(file_paths)
    size = len(acc_files)
    
    for i in range(size):
        gyro_file, ori_file = convert_acc_file_path(acc_files[i])
        ### !!!! ori_datalar textte timestamp-z-x-y seklinde yer aliyor !!! Diger dosyalardakilerinde de degistirmek lazim !!!
        ### read_csv'deki siralamayi timestamp z x y seklinde yapmak lazim
        ### ama benim resample edip kaydettigim dosyalarda x-y-z seklinde gidiyor oluyor
        acc_data = pd.read_csv(acc_files[i], header= None, index_col = 0, names = ["timestamp", "x", "y", "z"], skiprows = 16)
        gyro_data = pd.read_csv(gyro_file, header= None, index_col = 0, names = ["timestamp", "x", "y", "z"], skiprows = 16)
        ori_data = pd.read_csv(ori_file, header= None, index_col = 0, names = ["timestamp", "z", "x", "y"], skiprows = 16)        
        
        event_periods = get_event_periods(acc_files[i], gyro_file, ori_file)

        acc_freq = calculate_frequency(event_periods[0], acc_data.shape[0])

        if acc_freq >= min_acc_freq:
            print("Filename : ", acc_files[i])
            count = count + 1
            
            dataframe = resample_sensor_frequency_decimation(acc_data, gyro_data, ori_data,event_periods[0])
            
            if "ADL" in acc_files[i]: ### burasi daha efektif yapilmali yani bir kontrol yaptirtmadan ?
                file_path = acc_files[i].replace(FOLDER_PATH, COMBINED_DATA_PATH_ADL)
            else:
                file_path = acc_files[i].replace(FOLDER_PATH, COMBINED_DATA_PATH_FALL)
            
            file_name = file_path.split("/")[-1]            

            folder_path = file_path.replace("/" + file_name, "")
            file_name = file_name.replace("_acc_", "_")
            
            if not os.path.exists(folder_path):
                os.makedirs(folder_path) ### subdirectory'leri de create ediyor
            
            file_path = folder_path + "/" + file_name
            dataframe.to_csv(file_path, sep = '\t', index = False)

    print(">= 80Hz olan dosya sayisi: ", count)
    
    return   #### adl ve fall dosyaları ayri kaydedilmeli


### acc_freq/new_freq, gyro_freq/new_freq, ori_freq/new_freq => 2
### boyle olunca indis arttirimlari daha dogru oluyor sanki
### ilgili testler yapilmali, ayrica new_freq = 0 icin bir if kontrolu olmali
    
def resample_sensor_frequency_decimation_fixed(acc_data, gyro_data, ori_data,new_freq = 100):
    ### Normal sartlarda aslinda sample_duration tarzinda bir parametre gonderilmeli
    ### Ve frekanslar acc_freq = acc_data_size / sample_duration tarzinda bulunmali
        
    acc_freq = 100   ### Mobifall'daki datasetlerde 80Hz ve üstünü 100müs gibi dusunelim demistik
    gyro_freq = 200  ### Mobifall'daki datasetlerde gyroscope'un frekansı 200Hz'e çok yakın
    ori_freq = 200   ### Mobifall'daki datasetlerde orientation'un frekansı 200Hz'e çok yakın
    
    acc_increment  = acc_freq / new_freq
    gyro_increment = gyro_freq / new_freq
    ori_increment  = ori_freq / new_freq
        
    acc_values = extract_sensor_data(acc_data, acc_data.shape[0], acc_increment)
    gyro_values = extract_sensor_data(gyro_data, gyro_data.shape[0], gyro_increment)
    ori_values = extract_sensor_data(ori_data, ori_data.shape[0], ori_increment)
    
    ### print("---------   Before resizing -------")
    ### print_values(acc_values, gyro_values, ori_values)
    
    cut_some_observations(acc_values, gyro_values, ori_values)
    
    ### print("---------   After resizing -------")
    ### print_values(acc_values, gyro_values, ori_values)
    
    dataframe_data = get_dataframe_data_for_sensors(acc_values, gyro_values, ori_values)   
    dataframe = pd.DataFrame(data = dataframe_data)
    
    print(dataframe)
    
    return dataframe

"""
    Bu fonksiyonda acc,gyro ve ori'nin gercek frekansları alinark isleme tabii tutuldular ve 
    denk düşmeyen veriler için bir önceki ölçülen tekrar edilmiş oldu çünkü increment değeri 
    kesirli olduğu zaman integer'a çevrildiğinde 0.4 artış değeri varsa gerçek manada artış 3 adım sonunda
    gerçeklenmiş oluyor yani index0 = 0 ise inc=0.4 iken index0 = 0,0.4,0.8,1.2 şeklinde gidiyor ilk üç değer
    inte çevirilnce aynı indexe tekabül ediyor gyro ve oriden yeni değerler okunurken acc'den ise daha önceki 
    veriler tekrar edilmiş olarak resampling edilmiş oluyor
"""

def resample_sensor_frequency_decimation(acc_data, gyro_data, ori_data, event_period, new_freq = 100):
    ### Normal sartlarda aslinda sample_duration tarzinda bir parametre gonderilmeli
    ### Ve frekanslar acc_freq = acc_data_size / sample_duration tarzinda bulunmali
        
    acc_freq = calculate_frequency(event_period,acc_data.shape[0])   ### Mobifall'daki datasetlerde 80Hz ve üstünü 100müs gibi dusunelim demistik
    gyro_freq = calculate_frequency(event_period, gyro_data.shape[0])### Mobifall'daki datasetlerde gyroscope'un frekansı 200Hz'e çok yakın
    ori_freq = calculate_frequency(event_period, ori_data.shape[0])### Mobifall'daki datasetlerde orientation'un frekansı 200Hz'e çok yakın
    
    acc_increment  = acc_freq / new_freq
    gyro_increment = gyro_freq / new_freq
    ori_increment  = ori_freq / new_freq
        
    acc_values = extract_sensor_data(acc_data, acc_data.shape[0], acc_increment)
    gyro_values = extract_sensor_data(gyro_data, gyro_data.shape[0], gyro_increment)
    ori_values = extract_sensor_data(ori_data, ori_data.shape[0], ori_increment)
    
    ### print("---------   Before resizing ---------")
    ### print_values(acc_values, gyro_values, ori_values)
    
    ###acc_values, gyro_values, ori_values = cut_some_observations(acc_values, gyro_values, ori_values)
    cut_some_observations(acc_values, gyro_values, ori_values)
    
    ### print("---------   After resizing -------")
    ### print_values(acc_values, gyro_values, ori_values)
        
    dataframe_data = get_dataframe_data_for_sensors(acc_values, gyro_values, ori_values)
    
    dataframe = pd.DataFrame(data = dataframe_data)
    
    ### print(dataframe)
    
    return dataframe

"""  Bunlarda iste interpolation falan devreye girecekti ama yalan oldu elbette klasjdfklasjdf
def resample_with_numpy_fixed():
    return

def resample_with_numpy():
    return
"""

def get_dataframe_data_for_sensors(acc_values, gyro_values, ori_values):    
    dataframe_data = {
                      'acc-x' : acc_values['x'], 'acc-y' : acc_values['y'], 'acc-z' : acc_values['z'],
                      'gyro-x' : gyro_values['x'], 'gyro-y': gyro_values['y'], 'gyro_z' : gyro_values['z'],
                      'ori-x' : ori_values['x'], 'ori-y' : ori_values['y'], 'ori-z' : ori_values['z']
    }
        
    return dataframe_data

def extract_axis_values(sensor_data):
    sensor_values = {}
    
    sensor_values["x"] = sensor_data["x"].values
    sensor_values["y"] = sensor_data["y"].values
    sensor_values["z"] = sensor_data["z"].values
    
    return sensor_values
    
def extract_sensor_data(sensor_data, sample_size, increment):
    new_freq_data = {}
    
    new_freq_data["x"] = []
    new_freq_data["y"] = []
    new_freq_data["z"] = []
    
    sensor_values = extract_axis_values(sensor_data)
    
    index = 0.0    
    
    while int(index) < sample_size:
        new_freq_data["x"].append(sensor_values["x"][int(index)])        
        new_freq_data["y"].append(sensor_values["y"][int(index)])
        new_freq_data["z"].append(sensor_values["z"][int(index)])
        
        index = index + increment
    
    ### print("Sample-size : ", sample_size, "Increment: ", increment,"Index: ", index)
    return new_freq_data

def cut_some_observations(acc_values, gyro_values, ori_values):
    ### print("Cutting extra observations")
    
    acc_size = len(acc_values['x'])
    gyro_size = len(gyro_values['x'])
    ori_size = len(ori_values['x'])
    
    if acc_size < gyro_size and acc_size < ori_size:
        ### print("acc en kucuk")

        gyro_values['x'] = gyro_values['x'][ : acc_size]
        gyro_values['y'] = gyro_values['y'][ : acc_size]
        gyro_values['z'] = gyro_values['z'][ : acc_size]
        
        ori_values['x'] = ori_values['x'][ : acc_size]
        ori_values['y'] = ori_values['y'][ : acc_size]
        ori_values['z'] = ori_values['z'][ : acc_size]
        
    
    elif gyro_size < ori_size:
        ### print("gyro en kucuk")

        acc_values['x'] = acc_values['x'][ : gyro_size]
        acc_values['y'] = acc_values['y'][ : gyro_size]
        acc_values['z'] = acc_values['z'][ : gyro_size]
        
        ori_values['x'] = ori_values['x'][ : gyro_size]
        ori_values['y'] = ori_values['y'][ : gyro_size]
        ori_values['z'] = ori_values['z'][ : gyro_size]

    else:
        ### print("ori en kucuk")

        gyro_values['x'] = gyro_values['x'][ : ori_size]
        gyro_values['y'] = gyro_values['y'][ : ori_size]
        gyro_values['z'] = gyro_values['z'][ : ori_size]
        
        acc_values['x'] = acc_values['x'][ : ori_size]
        acc_values['y'] = acc_values['y'][ : ori_size]
        acc_values['z'] = acc_values['z'][ : ori_size]
    
    ### print_values(acc_values, gyro_values, ori_values)
    
    return
    
def convert_acc_file_path(acc_file):
    gyro_file = acc_path_to_gyro(acc_file)
    ori_file = acc_path_to_ori(acc_file)
    return gyro_file, ori_file

### event_period is in seconds, data_length is the number of observations
### so the return value of this function is in Hz
def calculate_frequency(event_period, data_length):
    frequency = data_length / event_period
    return frequency
    
def get_event_periods(acc_file, gyro_file, ori_file):
    event_periods = []
    
    acc_period = get_event_period(acc_file)
    gyro_period = get_event_period(gyro_file)
    ori_period = get_event_period(ori_file)
    
    event_periods.append(acc_period)
    event_periods.append(gyro_period)
    event_periods.append(ori_period)
    
    return event_periods

def get_event_period(file_name):
    ### skipping the unrelevant lines and reading the sample duration line
    with open(file_name) as file:
        skip_lines = 4
    
        for i in range(1, skip_lines + 1):
            file.readline()        
    
        period = file.readline() ### it has got unrelevant strings so we need to remove them
    period = formatting_event_period(period)
    
    return period

def formatting_event_period(period):
    #period = period.to_string()
    period = period.split("-")[-1]
    period = period.replace("s","")
    
    return int(period)

def print_values(acc_values, gyro_values, ori_values):

    print("size acc-x", len(acc_values['x']))
    print("size acc-y", len(acc_values['y']))
    print("size acc-z", len(acc_values['z']))
    print("size gyro-x", len(gyro_values['x']))
    print("size gyro-y", len(gyro_values['y']))
    print("size gyro-z", len(gyro_values['z']))
    print("size ori-x", len(ori_values['x']))
    print("size ori-y", len(ori_values['y']))
    print("size ori-z", len(ori_values['z']))    
    
    return

file_paths = get_files(FOLDER_PATH)
combine_sensor_files_2(file_paths)




