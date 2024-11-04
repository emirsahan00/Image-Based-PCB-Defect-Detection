# PCB Kusur Tespit

Bu proje, iki görüntü arasındaki farkları kullanarak endüstriyel kusur tespiti yapmayı amaçlamaktadır. Proje, OpenCV, NumPy, ve SciPy gibi kütüphanelerle birlikte, hata tespiti için SIFT özellik çıkarımı, FLANN eşleştirmesi, homografi hesaplama, IoU (Intersection over Union) metrikleri ve doğruluk değerlendirmesi içerir.

## Özellikler

- **SIFT Temelli Özellik Çıkarımı**: Görüntülerdeki anahtar noktalar çıkarılır ve tanımlayıcılar oluşturulur.
- **FLANN Eşleştirici**: Görüntülerdeki özellikler arasında eşleşme yapılır ve Lowe’un oran testi ile iyi eşleşmeler seçilir.
- **Homografi ve Hizalama**: İkinci görüntü, ilk görüntüye hizalanır ve fark görüntüsü çıkarılır.
- **Kusur Tespiti**: Farklı pikseller incelenir ve kusurlar tespit edilir.
- **IoU ve Doğruluk Değerlendirmesi**: Tespit edilen kusurlar, XML dosyasındaki gerçek kusur koordinatlarıyla karşılaştırılarak precision, recall, ve mAP gibi metrikler hesaplanır.
- **Görselleştirme**: Tespit edilen kusurlar kırmızı, gerçek kusurlar ise sarı renk ile görselleştirilir.

## Gereksinimler

Bu proje, aşağıdaki Python kütüphanelerini gerektirir:

- OpenCV
- NumPy
- SciPy
- xml.etree.ElementTree (Python yerleşik kütüphane)

Gerekli paketleri yüklemek için:

```bash
pip install opencv-python-headless numpy scipy
```
## Kullanım

Projenin kök dizininde, referans ve test görüntülerini içeren Reference/ ve rotation/ dizinlerini ve kusur koordinatlarını içeren Annotations/ dizinini oluşturun.  
Ana Python dosyasını çalıştırarak kusur tespitini ve doğruluk hesaplamasını başlatın.
```bash
defectDetection.py
```
Çalıştırma sırasında fark görüntüsü ve tespit edilen kusurlar görselleştirilecektir.

![hataGörüntüsü](https://github.com/user-attachments/assets/d80faa99-764f-45c5-a809-0b6ae7d8e582)


## Hesaplanan Metrikler

Precision: Doğru tespit edilen kusurların, toplam tespit edilen kusurlara oranı.  
Recall: Doğru tespit edilen kusurların, toplam gerçek kusurlara oranı.  
mAP (Ortalama Doğruluk): Precision-recall eğrisinin altındaki alanı ölçer.  
IoU: Her tespit edilen kusurun IoU değeri hesaplanır ve ortalama IoU sonucu görüntülenir.  

![precision](https://github.com/user-attachments/assets/911dcb0c-ef17-4e51-b54a-8a8c2afe8b96)








