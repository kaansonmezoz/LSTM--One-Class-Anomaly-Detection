Öncelikle /source içerisindeki combining-sensor-files.py çalıştırılmalıdır. Bu python dosyası farklı frekansta ve farklı dosyalarda olan sensor dosyalarını tek bir dosya içerisinde aynı frekansa sahip bir hale getirir. Ve bu dosyaları /resampled-data içerisinde toplar.

Ardından /source içerisindeki create_anomaly_detector.py dosyası çalıştırılmalıdır. Bu dosya anomali detector'ı oluşturur.

/data içerisinde raw olan Mobifall dataları bulunmaktadır.
/models içerisinde oluşuturulan modeller ve onlarla ilgili detaylar bulunur.

Bu dosya güncellenecektir.
Kullanılmayan fonksiyonlar, tekrar eden kod parçaları var .py dosyalarında, onlar da refactor edilecektir.

Notlar: 
* ADL'in başından kesmemek için read_from_file.get_samples(ADL_SET_PATH, OUTPUT_DIRECTORY + '/' + OUTPUT_MODEL_NAME + '/', 0, None, params['window_size'], params['sliding_window']) şeklinde çağırıldığında test_set_FALL'lar düzgün bir şekilde oluşmuyor. (0, 300, 9) şekline sahip bir dataframe oluşuyor. Bu hata çözülene kadar şimdilik oraların da kesilerek çağırılması gerekiyor.
* Sensör verilerinin 80Hz ve üstünde olanları sadece alınmıştır. Bunun altında kalan frekanslardaki sensör dosyaları resampling işlemine tabii tutulmayarak model için kullanılack verilerin içerisinde yer almamaktadır.
* MobiFall'ın readme dosyasında yer alan özellikler ile veri özellikleri arasında örtüşmeyen durumlar vardır. Fall olaylarının süresi 10 saniye olarak belirtilmiştir bazı fall olayları 10 saniyeden daha fazla sürmektedir. Sensör verilerinin örnekleme frekansı da readme yazan frekans ile farklılık göstermektedir.
* Kullanılan LSTM networkü "T. Theodoridis, V. Solachidis, Petros Daras, Nicholas Vretos" tarafından yayınlanan "Human Fall Detection from Acceleration Measurements Using a Recurrent Neural Network" makalesinden baz alınarak oluşturulmuştur.

-------> TODOS
  * Data preprocessing işlemlerinden normalizasyon eklenmeli --> Eklendi aslında ama değişiklik yapılması gerekebilir.
  * Confussion matrix, F1 score ... gibi değerler ve grafikler eklenmeli readme'ye
  * Refactoring 
    |-------------> create_LSTM_model.py : save_model ve save_parameters ile ilgili bir refactoring gerekebilir                 
    |-------------> create_anomaly_detector.py : Gereksiz yorum satirlari olabilir ilgilenmek gerekebilir bu durum ile.  
    |-------------> combining_sensor_files.py : Gereksiz yorum satirlari olabilir.  
    |-------------> read_from_file.py : Gereksiz yorum satirlari var onlar kaldirilmali. ismi files.py olmalı  
    |-------------> statistics.py : Gereksiz yorum satirlari var onlar kaldirilmali  
    |-------------> test_LSTM_model.py : Gereksiz yorum satirlari var onlar kaldirilmali. Daha iyi bir isim bulunabilir.  
    |-------------> visualization.py : Kod tekrarı var burasinin duzeltilmesi gerekiyor  
    
  * Tekrar eden fonksiyonlar, kod parçaları var 
  * Resampling frekansı ile ilgili değişiklikler yapılarak tekrardan denenebilir. Resampling metodu olarak numpy'in bir fonksiyonu vardı o kullanılarak tekrar denenebilir. Bununla ilgili yöntemi metodu seçen kod kısmı da yazılması gerekmektedir.
  * Daha detaylı bir dökümantasyon hazırlanmalıdır. Bu dökümantasyonda fonksiyonların yaptığı işler, aldıkları parametreler ve döndürdükleri sonuçlar yer almalıdır.
