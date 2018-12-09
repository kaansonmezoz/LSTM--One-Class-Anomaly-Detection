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
  * Data preprocessing işlemlerinden normalizasyon eklenmeli
  * Confussion matrix, F1 score ... gibi değerler ve grafikler eklenmeli readme'ye
  * Refactoring
  * Tekrar eden fonksiyonlar, kod parçaları var 
  * DataFrame'deki problem daha kalıcı bir şekilde çözülmeli. Problem de yeni bir dataframe oluşturmak yerine bellekte ilk oluşturana ekleniyor dolayısıyla ADL ve FALL'lar aynı dataframe'de oluyor. Değişkenler temizlenmeden belli parçaları kodun tekrardan çalıştırıldığında eklemeye devam ediyor aynı dataframe'ye bu da tekrar eden windowların train set, test set gibi setlerde oluşmasına neden oluyor. Bu da hatalı sonuca yol açıyor. Dolayısıyla tekrardan bütün kodun çalıştırılması ve değişkenlerin temizlenmesi (Spyder ---> remove variables gibi ) gerekiyor.
  * Hareketlerden saniyeleri atma biraz problemli çalışıyor. Şöyle ki ADL'den herhangi bir atma yapmak istemezsek ve bunun için ilgili değerleri verirsek FALL'larda herhangi bir veri oluşmuyor okurken dosyadan. Burada bir problem var bu çözülmeli. Bu yüzden şimdilik hem fall'dan hem de adl verilerinin başından ve soununda kesme yapılıyor.
  
