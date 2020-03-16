# data-generation
Synthetic data generation from a single sample. 

# Overview
Bu sürüme kadar sol beyin lobu verilerinden train ve evaluation olmak üzere 2 tane veriseti oluşturuldu.
Bu veri setlerinin içindeki deneklerin cross-distance uzaklıkları hesaplandı. Bu sayede bir veri kümesi tamamen kendisinin aynısı gibi üretildiğinde cross-distance degeri ne olur o görülmüş oldu.
Sonrasında matlab kütüphanesinde yer alan mvnrnd() multivariate random distribution metodu kullanılarak sentetik veri üretimi yapıldı. Bu işlem yapılırken iki degiskene ihtiyac duyuldu. bunlardan bir tanesi oluşturulacak sentetik datasetin mean ve sigma degeri. mean degeri zaten mevcut olan CBT verieleri kullanılarak elimizde zaten mevcuttu. Sigma degeri ise asıl bulmamız gereken bilinmeyen degerdi. 

## Step 1
Sigma array'indeki tum degerler 1 kabul edilerek bir geliştirme yapıldı. Bu geliştirme sonucunda üretilen dataset ile evaluation set arasında cross-distance hesaplaması yapıldı ve işlemin başarısı ölçüldü.

## Step 2
Sigma degere random number generator ile üretildi ve bu sigma degeri kullanılarak yeni sample'lar elde edildi. Eldde edilen set yine evaluation set ile cross-distance hesabına sokuldu ve sonuçlar ölçüldü.

## Step 3 
Sonrasında farklı bir yöntem denenerek sigma degeri train set içerisndeki deneklerin sigma degeri hesaplanarak yapıldı. Bu sonuçta evaluation set ile cross-distance hesabı yapıldı. Bu ölçümde alınan sonuç en başarılı sonuçtu. Bunun sebebi sigma degerinin orjinal degerden elde edilmesi. Bir bakıma buradan elde edilen sonuç elde edilebilecek maksimum sonuç olarak gösterilebilir. Ama bu yöntem normal bir yöntem olarak kullanılamaz, çünkü algoritma kullanılırken orjinal veriler elimizde hiçbir zaman olmayacak.

## Step 4 
Bu adımda sigma degerini optimize etmek için genetik algoritma kullanıldı. Rastgele 60 tane sigma degeri oluşturulup, bu degerler ile ayrı ayrı sentetik veri kümeleri oluşturuldu. Herbir veri kümesinin evaluation kümesi ile cross-distance hesabı yapıldı. Bu hesap sonucunda en güzel sonucu veren belli miktardaki sigma degerleri seçildi. Seçilen bu degerler genetik algoritma ile crossover ve mutasyon işlemlerine sokuldu. Bu işlem sonucunda elde edilen children sigma degerleri ile tekrardan sentetik veri setleri üretildi ve evaluation set ile cross-distance hesapları yapıldı. Bu hesaplamalar sonucunda yeni üretilen children sigmalar ve eski sigmalar karşılaştırılarak en başarılı olanlar seçildi ve crossover mutasyon işlemleri belli bir  başarı elde edilene kadar tekrarlandı. Bu işlem sonucunda sigma degerlerinin 3. adımdaki başarıya eriştigini çok küçük bir fark ile geçtiği gözlemlendi.


![Step-4 Iteration Graph](https://github.com/celikmustafa89/data-generation/tree/develop/figures/iteration-graph1.png)

![General Comparison Graph](https://user-images.githubusercontent.com/6848680/76770008-4eaaac80-67ae-11ea-974e-5473e34fc441.png)
