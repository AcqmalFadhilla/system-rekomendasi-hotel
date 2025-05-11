# Laporan Proyek Machine Learning-Muh. Acqmal Fadhilla Latief

## Project Overview

Seperti kita ketahui bahawa sistem rekomendasi sangat di butuhkan dalam dunia *bussines* karena dapatkan meningkatkan penjualan,saya membuat sistem rekomendasi hotel 

seperti penjelasan di atas bahwa sistem rekomendasi dapatkan meningkatkan penjualan,dikarenakan dapat merokemendasikan *user* hotel yang belum mereka tuju dengan pelayanan yang sesuai *user* inginkan

## Business Understanding

Berdasarkan kondisi yang telah di uraikan pada *Project* *overview*

### Problem Statements

1. Bagaimana membuat sistem rekomendasi hotel berdasarkan kota,r*ating* hotel dan keinginaan user
2. Bagaimana membantu *manager* hotel untuk mengetahui apakah hotelnya termasuk paling rekomendasi atau tidak

### Goal

1. Bisa merekomendasikan hotel sesaui dengan kota,*rating* hotel dan keinginaan *user*
2. Bisa merekomendasikan hotel-hotel baru sesuai dengan keinginaan *user*

## Data Understanding

Dataset saya dapatkan dari kaggle yaitu [hotel recommendation](https://www.kaggle.com/datasets/keshavramaiah/hotel-recommendation?select=Hotel_Room_attributes.csv) dengan banyak sekali missing value jadi perlu melakukan data preparation

saya memakai 3 dataset  dan jumlah banyak datanya yaitu

1. Hotel_details.csv = 99154
2. Hotel_Room_attributes.csv = 161054
3. hotels_RoomPrice.csv = 165873

Penjelasan kolom dataset hotel_details.csv: 

- id = number yang unik
- hotel id = id hotel
- name_hotel = nama hotel
- address = alamat
- city = city(terutama dari hotel eropa)
- country =negara
- zipcode =kode pos
- propertytype = (hotel, resor, taman hiburan, dll)
- starrating = ini adalah peringkat pengguna ini bukan yang stabil.
- Latitude  = untuk penandaan geografis

Penjelasan kolom dataset Hotel_Room_attributes.csv:

- id = number unik
- hotelcode = kode hotel
- roomamenities = fasilitas kamar yang ditawarkan hotel
- roomtype = fasilitas kasur yang di tawarkan oleh hotel
- ratedescription = penjelasan lebih lanjut tentang fasilitas hotel

Penjelasan kolom dataset hotel_roomPrice.csv:

- id = number unik
- refid = number unik pembayaran
- hotelcode = kode hotel
- dtcollated = tanggal pengambilan data
- ratedate = tanggal pembayaran
- guest = jumlah tamu
- lost = jumlah tamu yang keluar
- roomtype = tamu memesan kamar type apa
- onsirate = uang yang di bayarkan oleh tamu untuk mendapatkan fasilitas dan kamar tersebut

## Data Preparation

### Missing values

sudah saya sebutkan pada data understanding bahwa dataset yang saya pakai memiliki banyak sekali *missing value,*dengan detail sebagai berikut:

- address = 5093
- url = 111
- roomamenities = 4819
- ratedescription = 4819
- ratedescription = 4819
- ratetype = 4819
- promoname = 162696
- taxtype = 8370
- mealinclusiontype = 68486
- hotelblock = 1610554

cara saya mengatasi *missing value* tersebut dengan menggunakan method dropna()

cara saya mengatasi *missing value* tersebut dengan cara mendrop *missing value* tersebut karna saya beranggapan bahwa missing value tersebut tidak akan mengangu kinerja dari model nantinya maka dari saya melakukan drop untuk *missing value* karna data yang saya miliki cukup banyak

### Drop

selain itu juga saya melakukan *drop* pada kolom yang menurut saya tidak memiliki relasi yang kuat untuk meningkatkan kinerja model,adapun kolom yang saya *delete* yaitu:

- id
- zipcode
- hotelid
- url
- curr
- Source

selain melakukan *drop* pada kolom saya melakukan drop pada data duplikat pada kolom hotelid di dataset hotel details

## Marge

Selain itu saya juga melakukan *marge* pada dataset hotel room dan hotel details dengan acuan kolom hotel code dan hotel id,dengan cara memakai salah tau method *library* pandas yaitu *marge*

## Model

system rekomendasi saya buat berdasarkan rating dan kota di mana N-top nya adalah rating dari hotel tersebut berdasarkan kota hotel dan roomtype dimana semakin bagus room type yang di berikan pada hotel tersebut maka semakin rekomendasi dan akan muncul pada rekomendasi paling atas

untuk itu mencapai itu saya memberikan value number pada roomtype dari 1-4,semakin besar value nya maka semakin bagus:

```python
room_no=[
     ('king',2),
   ('queen',2), 
    ('triple',3),
    ('master',3),
   ('family',4),
   ('murphy',2),
   ('quad',4),
   ('double-double',4),
   ('mini',2),
   ('studio',1),
    ('junior',2),
   ('apartment',4),
    ('double',2),
   ('twin',2),
   ('double-twin',4),
   ('single',1),
     ('diabled',1),
   ('accessible',1),
    ('suite',2),
    ('one',2)
   ]
```

seperti yang saya bilang di pejelasan sebelumnya semakin besar valuenya maka semakin rekomendasi pula hotel tersebut

jadi ada tiga unsur rubrik penilaian system rekomendasi hotel model yaitu kota,rating,dan roomtype,

| Rubrik penilaian | penjelasan |
| --- | --- |
| kota | Dimana hotel berada misalnya london |
| rating | rating di ambil dari kolom peringkat dari 1-4 |
| roomtype | jenik kamar type apa yang di inginkan oleh user |

outputnya sebagai berikut:

| Nama Hotel | room type | starrating | guset | roomamenities |
| --- | --- | --- | --- | --- |
| J Hotel London | Executive Triple | 4 | 3 | air conditioning,blackout curtains,coffee/tea . |
| Copthorne London Gatwick | Superior Triple Room | 4 | 3 | air conditioning,alarm clock,carpeting,closet, |
| Hotel Henry VIII | Triple | 4 | 3 | air conditioning,blackout curtains,closet, |
| Barrington Lodge | Deluxe Triple Room | 4 | 3 | air conditioning,alarm clock,carpeting,closet |
| Lucky 8 Hotel | Triple Room | 4 | 3 | air conditioning,free wi-fi in all rooms! |

Adapun model yang saya buat termasuk algoritma content based filtering di karenakan merekomendasikan item yang mirip dengan item yang disukai pengguna di masa lalu.

## Evaluasi

Seperti yang saya sudah jelas kan pada model,bahwa saya menggunakan algortima content based filtering  di karenakan merekomendasikan item yang mirip dengan item yang disukai pengguna di masa lalu.

saya membuat evaluasi berdasarkan model dan permintaan *feature* oleh *user,*dimana *feuture* ini saya ambil dari valuel kolom city,hotelname,roomtype,gusetno,starrating,city,address,roomaminities,ratesdescription,dan similarity dengan bantuan library nltk untuk membantuk deteksi dari kata yang di inginkan oleh userrekomendasi('London', 4, 'I need hot water.')

```python
rekomendasi('London', 4, 'I need hot water.')
```

output code di atas akan merekomendasikan

| Hotel Name | room type | guset no | starrating | roomamenities | ratedescription | similarity |
| --- | --- | --- | --- | --- | --- | --- |
| Best Western Swiss Cottage | Family Room | 4 | 3 | air conditioning,alarm clock,carpeting,closet | Room size: 25 m²/269 ft², Non-smoking, Shower | 1 |
| The Lord Lister Hotel | Family Room | 4 | 3 | air conditioning,alarm clock,carpeting,closet | Room size: 21 m²/226 ft², Shower and bathtub | 1 |
| Hallmark Hotel London Chigwell Prince Regent | Executive Family Room | 4 | 3 | air conditioning,carpeting,closet,clothes | Non-smoking, Shower and bathtub, 1 double bed | 1 |
| The Park City Grand Plaza Kensington Hotel | Family Suite | 4 | 3 | air conditioning,alarm clock,blackout curtains | Room size: 31 m²/334 ft², City view, Non-smoki | 1 |

adapun metrik yang saya gunakan adalah metrik presisi yaitu mencari kesamaan item dengan item yang di inginkan oleh *user*

jadi model harus merekmendasikan item yang relevan kepada pengguna,sistem rekomendasi menggunkan metrik presisi menilai setiap item kandidat sesuai dengan metrik kesamaan,

contoh case:

Any ingin memesan sebuah hotel di london dengan fitur memliki air condioning,alaram clock dan memiliki closet,jadi sistem akan me rekomendasikan hotel apa?

| hotel name | city | roomanties | roomanties | roomanties | rating |
| --- | --- | --- | --- | --- | --- |
|  |  | air conditioning | alarm clock | closet |  |
| the lord lister hotel | Italy | 1 | 1 | 1 | 4 |
| J hotel london | london | 1 | 0 | 0 | 4 |
| hotel henrey VII | london | 1 | 1 | 1 | 4 |
| lucky 8 hotel | london | 0 | 1 | 0 | 4 |
| Regency House Hotel | london | 0 | 0 | 0 | 4 |

| Nama user | ciity | roomanties | roomanties | roomanties |
| --- | --- | --- | --- | --- |
| any | london | 1 | 1 | 1 |
| hotel henrey VII | 1 | 1 | 1 | 1 |
| lucky 8 hotel | 1 | 0 | 1 | 0 |
| ucky 8 hotel | 1 | 0 | 1 | 0 |

jika di lihat dari tabel di atas maka sistem rekomendasi akan merekomendasikan hotel henrey VIII

karna kesamaan item yang di inginkan oleh user dan item kandidat karna memeliki nilai yang paling dekat dengan item yang di inginkan oleh pengguna dengan nilai 1 di anggap paling mendekati dan 0 dianggap paling tidak mendekati oleh keinginaan pengguna.begitulah klira-kira kerja dari metrik  presisi