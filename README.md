Öncelikle /source içerisindeki combining-sensor-files.py çalıştırılmalıdır. Bu python dosyası farklı frekansta ve farklı dosyalarda olan sensor dosyalarını tek bir dosya içerisinde aynı frekansa sahip bir hale getirir. Ve bu dosyaları /resampled-data içerisinde toplar.

Ardından /source içerisindeki create_anomaly_detector.py dosyası çalıştırılmalıdır. Bu dosya anomali detector'ı oluşturur.

/data içerisinde raw olan Mobifall dataları bulunmaktadır.
/models içerisinde oluşuturulan modeller ve onlarla ilgili detaylar bulunur.

Bu dosya güncellenecektir.
Kullanılmayan fonksiyonlar mevcuttıur .py dosyalarında onlarda refactor edilecektir.
