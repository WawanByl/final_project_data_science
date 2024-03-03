# Penentuan Biaya Asuransi Berdasarkan Profil Klien

<img src="https://www.python.org/static/community_logos/python-logo.png" alt="icon python" title="Judul Gambar" width="300" height="100">

## Table of Contents
[1. Latar Belakang](https://github.com/WawanByl/final_project_data_science/edit/main/README.md#1-latar-belakang)
[2. Tujuan](https://github.com/WawanByl/final_project_data_science/edit/main/README.md#2-tujuan)
[3. Data Understanding](https://github.com/WawanByl/final_project_data_science/edit/main/README.md#3-data-understanding)
[4. Exploratory Data Analysisi](https://github.com/WawanByl/final_project_data_science/edit/main/README.md#4-exploratory-data-analysis)
[5. Permodelan](https://github.com/WawanByl/final_project_data_science/edit/main/README.md#5-permodelan)
[6. Kesimpulan](https://github.com/WawanByl/final_project_data_science/edit/main/README.md#6-kesimpulan)
[7. Appendix](https://github.com/WawanByl/final_project_data_science/edit/main/README.md#7-appendix)

### 1. Latar Belakang
1. Dalam bisnis asuransi, ada beberapa poin yang perlu diperhatikan salah satunya adalah penentuan premi,  manajemen risiko, dan personalisasi penawaran.
2. Penentuan Premi yang lebih akurat sangat diperlukan untuk menilai risiko individu berdasar factor-factor; usia, BMI, status perokok atau bukan, jumlah anak dan tempat tinggal.
3. Manajemen Risiko yang lebih baik dapat membantu perusahaan dalam mengidentifikasi polis yang berpotensi tinggi, dan mengambil langkah-langkah preventif.
4. Optimasi Strategi Pemasaran, analisis data yang mendalam memungkinkan Perusahaan mengidentifikasi  segmen pelanggan yang potensial dan menargetkan mereka dengan marketing  
   campaign yang efektif

### 2. Tujuan
1. Tujuan kita kali ini adalah untuk memprediksi berapa charges (biaya asuransi) berdasarkan variable lain seperti; age, BMI, children, sex, smoker dan region.
2. Mendapatkan daftar fitur & pentingnya ( feature importances) dari model terbaik yang nantinya kita pilih
3. Menentukan model yang digunakan, apakah model regresi atau klasifikasi?

### 3. Data Understanding
insurance.csv berisi kolom antara lain :
age: Usia penerima asuransi.
bmi: Body Mass Index, memberikan pemahaman tubuh, berat yang dianggap sehat atau tidak sehat berdasarkan tinggi.
children: Jumlah anak yang ditanggung oleh asuransi.
smoker: Apakah penerima asuransi perokok atau bukan.
region: Wilayah tempat tinggal penerima asuransi di AS (northeast, southeast, southwest, northwest).
charges: Biaya asuransi kesehatan individu.

### 4. Exploratory Data Analysis
![image](https://github.com/WawanByl/final_project_data_science/assets/153416221/b56e43c4-bca1-436b-9372-733807f1ebd2)

![image](https://github.com/WawanByl/final_project_data_science/assets/153416221/0b0f4422-611d-48ff-b457-263c3b0f5e67)

![image](https://github.com/WawanByl/final_project_data_science/assets/153416221/5e6b9ef8-8aa5-4a0f-b3f4-50119aa70fdf)

![image](https://github.com/WawanByl/final_project_data_science/assets/153416221/e7b06809-925a-4b0a-8301-2cde3c15a3b1)

![image](https://github.com/WawanByl/final_project_data_science/assets/153416221/4dbc1ca6-48e9-4250-97bd-a7557108b174)

![image](https://github.com/WawanByl/final_project_data_science/assets/153416221/7bca6ec6-63ff-466e-8b9d-556e5ec67c23)

![image](https://github.com/WawanByl/final_project_data_science/assets/153416221/72d2ae33-4623-4806-9134-46066b75ce71)

Exploratory Data Analysis mengungkapkan beberapa insight:

Matriks Korelasi: Usia dan BMI menunjukkan korelasi positif dengan biaya asuransi, meskipun moderat. Ini menunjukkan bahwa seiring bertambahnya usia dan BMI, biaya asuransi juga meningkat. Jumlah anak memiliki korelasi yang sangat rendah dengan biaya.

Pairplot dengan Status Merokok: Status merokok memiliki dampak signifikan terhadap biaya asuransi, seperti yang terlihat pada scatter plot. Perokok cenderung memiliki biaya yang lebih tinggi di semua usia dan BMI dibandingkan dengan non-perokok.

Biaya Asuransi berdasarkan Jumlah Anak: Jumlah anak tidak menunjukkan tren yang jelas dengan biaya asuransi. Namun, individu dengan 3 anak atau lebih cenderung memiliki rentang biaya asuransi yang lebih luas.

Biaya Asuransi berdasarkan Jenis Kelamin: Terdapat perbedaan kecil dalam biaya antara laki-laki dan perempuan, dengan laki-laki umumnya menghadapi biaya yang lebih tinggi.

Biaya Asuransi berdasarkan Wilayah: Tidak tampak adanya perbedaan signifikan dalam biaya berdasarkan wilayah, menunjukkan bahwa lokasi mungkin bukan faktor utama dalam menentukan biaya asuransi.

### 5. Permodelan
Modeling Data : Model sederhana ke model yang lebih kompleks

Regresi linear sederhana
Model regresi sederhana dalam machine learning adalah metode yang digunakan untuk memahami dan memprediksi hubungan antara satu variabel independen (biasanya disebut sebagai variabel prediktor) dan satu variabel dependen (biasanya disebut sebagai variabel target). Ide dasarnya adalah menggunakan data yang ada untuk membangun persamaan matematis yang menggambarkan hubungan antara variabel prediktor dan variabel target.

Predictive Analysis menggunakan model regresi linier sederhana menghasilkan kesalahan kuadrat rata-rata (MSE) sekitar 3.589479e+07 dan skor R2 sebesar 0.80. Skor R2, yang berkisar dari 0 hingga 1, menunjukkan bahwa sekitar 80% variabilitas dalam biaya asuransi dapat dijelaskan oleh model. 

Untuk membuat model machine learning yang lebih kompleks dan mengevaluasinya dengan mempertimbangkan baseline model, kita akan mengambil beberapa langkah. Pertama, kita akan menggunakan model regresi linier sebagai baseline, yang sudah kita buat sebelumnya. Kemudian, kita akan mencoba model yang lebih kompleks, seperti Random Forest, XGBoost, SVM, dll untuk melihat apakah kita dapat meningkatkan performa model dibandingkan dengan baseline.
Langkah-langkahnya sebagai berikut:
Model Baseline: Model regresi linier yang sudah kita buat dan evaluasi dengan R2 score sekitar 0.80 akan dijadikan sebagai baseline.
Model Kompleks: Akan menggunakan Random Forest Regressor, XGBoost, SVM, dll
Evaluasi Model: Kita akan membandingkan performa kedua model tersebut menggunakan R2 score dan Mean Squared Error (MSE) sebagai metrik evaluasi.

![image](https://github.com/WawanByl/final_project_data_science/assets/153416221/f55576d5-cf92-45b1-a44f-8a6d0dfc10d1)

Predictive Analysis terbaik menggunakan model Random Forest Regressor menghasilkan kesalahan kuadrat rata-rata (MSE) sekitar 21196800.078058 dan skor R2 sebesar 0.88. ini menunjukkan bahwa sekitar 88% variabilitas dalam biaya asuransi dapat dijelaskan oleh model. Ini menunjukkan bahwa model tersebut, dengan mempertimbangkan usia, BMI, status perokok, dan faktor lainnya, cukup efektif dalam memprediksi biaya asuransi.

Sebelum melangkah lebih jauh, coba kita cek data diatas dengan hasil evaluasi data training untuk cek apakah model kita itu sudah fit, overfitting, atau underfitting?

![image](https://github.com/WawanByl/final_project_data_science/assets/153416221/dbb86c8c-096c-4b72-b884-4c54d036b060)

Hasil evaluasi data train RandomForestRegressor menghasilkan MSE = 3511982.408 dan skor R2 sebesar 0.97. Dibandingkan data test MSE = 21196800.078 dan skor R2 sebesar 0.88 
ini menunjukkan adanya penurunan performa model, pada data train hasilnya lebih baik dibandingkan data test, ini menunjukkan bahwa model ada indikasi overfitting.

Langkah selanjutnya adalah tuning terhadap model

Hasil evaluasi data test RandomForestRegressor sebelum tuning menghasilkan MSE = 21196800.078 dan skor R2 sebesar 0.88 
Setelah tuning diperoleh data test MSE = 19382443.512 dan skor R2 sebesar 0.89 
Peningkatan nilai R2 dan penurunan MSE pada data test setelah 'tuning' menunjukkan bahwa model menjadi lebih akurat dan konsisten dalam prediksinya terhadap data test.

Kesimpulan Mengingat tidak ada perubahan signifikan pada data train dan terjadi perbaikan pada data test, proses 'tuning' yang kita lakukan berhasil mengurangi overfitting tanpa menyebabkan underfitting. 

Model sekarang lebih baik dalam menggeneralisasi dari data train ke data test, yang merupakan tujuan utama dari proses 'tuning'. Ini menunjukkan bahwa model ini sekarang berada dalam kondisi just fit, dimana model memiliki keseimbangan yang baik antara kemampuan untuk menangkap tren dalam data train dan kemampuan untuk generalisasi pada data baru.

Kesimpulan Evaluasi Model: Model Random Forest Regressor menunjukkan peningkatan signifikan dalam performa dibandingkan dengan model regresi linier, yang dijadikan sebagai baseline. Ini ditunjukkan melalui penurunan MSE dan peningkatan R2 score. Random Forest Regressor, dengan kemampuannya untuk menangkap non-linearitas dan interaksi antar variabel, memberikan prediksi yang lebih baik untuk biaya asuransi berdasarkan faktor-faktor seperti usia, BMI, status perokok, dan lainnya. Ini menggarisbawahi pentingnya memilih model yang tepat untuk data dan masalah yang spesifik. Meskipun model yang lebih kompleks seperti Random Forest dapat memberikan performa yang lebih baik.

![image](https://github.com/WawanByl/final_project_data_science/assets/153416221/78c7cafb-db74-425b-aca3-086fba6f7d4e)

Berikut adalah daftar fitur dan pentingnya (feature importances) dari model Random Forest Regressor, yang menunjukkan seberapa penting setiap fitur dalam memprediksi biaya asuransi:

smoker_0 dan smoker_1 (dengan nilai penting sekitar 0.49 dan 0.14) menunjukkan status perokok sangat berpengaruh terhadap prediksi biaya asuransi. Ini mengindikasikan bahwa apakah seseorang merokok atau tidak adalah faktor penting.

bmi memiliki nilai penting sekitar 0.19, menandakan bahwa indeks massa tubuh (BMI) juga berpengaruh besar terhadap biaya asuransi.

age dengan nilai penting sekitar 0.14, menunjukkan usia sebagai faktor penting lainnya.

Fitur lain seperti region_1, children, region_0, region_2, sex_0, sex_1, dan region_3 memiliki nilai penting yang lebih rendah, menandakan bahwa mereka memiliki pengaruh yang lebih kecil terhadap prediksi biaya asuransi.

Fitur smoker_0 dan smoker_1 adalah hasil dari pengkodean one-hot untuk variabel smoker, di mana merepresentasikan perokok dan yang lainnya non-perokok. 

Hal serupa berlaku untuk fitur lain yang dihasilkan dari pengkodean one-hot. Dengan informasi ini, Anda dapat melihat bagaimana setiap fitur berkontribusi terhadap prediksi model

Model machine learning yang telah kita kembangkan dan evaluasi untuk memprediksi biaya asuransi dapat memiliki implikasi signifikan dalam konteks bisnis asuransi, terutama dalam hal penentuan premi, manajemen risiko, dan personalisasi penawaran. Berikut ini adalah beberapa cara model ini dapat diimplementasikan dalam skenario bisnis nyata dan dampak yang ditimbulkannya:

1. Penentuan Premi yang Lebih Akurat
- Implementasi: Model dapat digunakan untuk menilai risiko individu lebih akurat berdasar faktor-faktor ; usia, BMI, status perokok, jumlah anak, dan wilayah tempat tinggal. Dengan memperhitungkan variabel- 
  variabel ini, perusahaan dapat menentukan premi yang lebih sesuai dengan risiko sebenarnya.
- Dampak: Hal ini akan memungkinkan perusahaan asuransi untuk lebih adil dalam menetapkan premi, mengurangi subsidisilintas antar pelanggan, dan pada akhirnya meningkatkan kepuasan pelanggan. Ini membantu 
  mengoptimalkan margin keuntungan dengan mengurangi eksposur klaim berisiko tinggi.

 2. Manajemen Risiko yang Lebih Baik
- Implementasi: Model dapat membantu perusahaan dalam mengidentifikasi polis yang berpotensi berisiko tinggi dan mengambil langkah-langkah preventif, seperti menawarkan program kesehatan dan kebugaran untuk 
  individu dengan BMI tinggi atau program berhenti merokok.
- Dampak: Dengan mengurangi risiko, perusahaan asuransi dapat menurunkan jumlah klaim besar dan memperbaiki kinerja keuangan secara keseluruhan. Ini juga membantu membangun reputasi positif di pasar sebagai 
  perusahaan yang peduli dengan kesehatan dan kesejahteraan pelanggannya.

 3. Personalisasi Penawaran
- Implementasi: Menggunakan model untuk menggali insight tentang kebutuhan dan risiko pelanggan memungkinkan perusahaan menyesuaikan produk dan layanannya sesuai karakteristik individu.
- Dampak: Personalisasi ini meningkatkan penjualan dan retensi pelanggan dengan menawarkan polis yang relevan dan menarik bagi pelanggan. Ini memberi kesempatan berinovasi dalam produk dan layanan yang 
  ditawarkan.

4. Optimisasi Strategi Pemasaran
- Implementasi: Analisis data yang mendalam memungkinkan perusahaan mengidentifikasi segmen pelanggan yang potensial dan menargetkan mereka dengan kampanye pemasaran yang efektif.
- Dampak: Dengan fokus pada pelanggan yang paling mungkin tertarik dengan produk asuransi tertentu, perusahaan dapat meningkatkan ROI dari pengeluaran pemasarannya dan efisien mengalokasikan sumber dayanya.

### 6. Kesimpulan
Implementasi model prediktif dalam bisnis asuransi tidak hanya membantu dalam penentuan premi yang akurat dan manajemen risiko tetapi juga meningkatkan pengalaman pelanggan melalui personalisasi penawaran dan komunikasi. Meskipun demikian, penting untuk terus menguji dan memperbarui model dengan data terbaru untuk memastikan akurasi dan relevansinya, serta mempertimbangkan aspek etika dan privasi dalam penggunaan data pelanggan.

### 7. Appendix
https://www.kaggle.com/datasets/mirichoi0218/insurance
