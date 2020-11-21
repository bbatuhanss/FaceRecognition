6
"""
Created on Thu Jun  4 00:36:03 2020

@author: Batuhan Sevinç
"""

import cv2
import numpy as np
from PIL import Image
import os
cam = cv2.VideoCapture(0)
cam.set(3, 640) # video genişliğini 
cam.set(4, 480) # video yüksekliğini
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_id = 1#  kişi için bir sayısal yüz kimliği değeri atama
print("\n Yüz yakalama başlatılmıştır.Lütfen kameraya bakınız.")
#Bireysel örnekleme yüz sayısını başlatma kodu aşağıdadır.
count = 0
while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)# Tanımalanan yüzün etrafında yeşil bir kare oluşturuluyor
        count += 1
        # Çekilen görüntüyü veri kümeleri klasörüne kaydediliyor
        cv2.imwrite("dataset/User" + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('image', img)# Sonuç ekranda gösteriliyor.
    k = cv2.waitKey(100) & 0xff 
    if k == 27:
        break

    elif count >= 15: # 15 yüz örneği alınıyor ve video durdurulyor
         break

# herşey tamamsa ekran yakalaması serbest bırakılıyor.
cam.release()
cv2.destroyAllWindows()
path = 'dataset' # Yüz görüntülerinin kaydedildiği veri klasörümüz
recognizer = cv2.face.LBPHFaceRecognizer_create()
# fonksiyonunumuz  görüntüleri ve etiket verilerini almak için kullanılmaktadır.
def getImagesAndLabels(path):

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]#ImagePaths listesi, path klasörü içinde yer alan dosyaların listesidir.  
    faceSamples=[]#faceSamples listesi yüz görüntülerini tutacak.
    ids = []#ids listesi  her bir görüntünün etiketini tutacak.
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L') #imagePaths listesindeki her bir elemanı kullanrak veri klasörümüzden aldığımız fotoları gri tonlamaya dönüştüren kod.
        img_numpy = np.array(PIL_img,'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])# dosya adından kişi numarasını ayırıp, bir tam sayı olarak id değişkenine atıyoruz.
        faces = face_detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples,ids

print ("\n Eğitim yüzleri işleniyor. Bekleyiniz...")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids)) #veri kümemizi eğitiyoruz.
recognizer.write('trainer/trainer.yml') #Modelimizi  trainer / trainer.yml içine kaydettik.


print("\n {0} tane eğitilmiş yüz bulunmaktadır.".format(len(np.unique(ids)))) #Eğitimli yüzlerin sayısını yazdırdık
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')#Yüz tanıma nesnemiz recognizer‘a trainer.yml dosyası aracılığıyla eğitilmiş verisetimizi yüklüyoruz 
font = cv2.FONT_HERSHEY_SIMPLEX

id = 0 #kimlik sayacımızı oluşturduk
names = ['None', 'Batuhan'] # id ye karşılık gelen adlar (Bu kişi tanıtma birden daha fazla olabilir ama ben tek kişi olarak yaptım.)
cam = cv2.VideoCapture(0)# Gerçek zamanlı video yakalamay  başlatmayı sağlar
cam.set(3, 1000) # video genişliğini 
cam.set(4, 800) # video yüksekliğini

# Yüz olarak tanınacak pencere boyutu ayarlama 
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)
print("\n Çıkmak için q ya basınız!")

while True:
    ret, img =cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )# görüntü karesindeki tüm yüzleri yakalıyor.

    for(x,y,w,h) in faces: #yüz dikdörtgenlerini bir döngü içinde işleme sokuyoruz.
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])#recognizer.predict() metodu id ve confidence değerlerini döndürüyor. id kişi numarası; confidence ise yapılan saptamanın tahmini doğruluk oranıdır  
        if (confidence < 100):
            id = names[1]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        
        cv2.putText(img, str(id), (x+5,y-25), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    
    cv2.imshow('camera', img)#Görüntünün ekrana yansıtılması.
    k = cv2.waitKey(10) & 0xff 
    if k == 27 or k==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
