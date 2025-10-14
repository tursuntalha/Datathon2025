## Datathon2025

Bu repo, Datathon 2025 yarışması için hazırlanan veri analizi ve modelleme çalışmalarını içerir. Çalışma Jupyter Notebook üzerinde gerçekleştirilmiş, özellik mühendisliği ve LightGBM tabanlı bir regresyon modeli ile oturum değerinin (session_value) tahminlenmesi hedeflenmiştir.

### Proje Yapısı
- `datathon2025.ipynb`: Tüm veri hazırlama, özellik çıkarımı, modelleme ve tahmin adımlarını içeren notebook.
- `datathon_dataset/`: Yerel veri klasörü.
  - `train.csv`
  - `test.csv`
  - `sample_submission.csv`
- `outputs/`: Çalışma sonucu üretilen dosyalar (örn. submission) burada oluşturulur.

### Kurulum
1. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```
2. Notebook'u açın ve çalıştırın:
```bash
jupyter notebook datathon2025.ipynb
```

### Veri Kaynağı ve Atıf
Bu projede kullanılan veriler, Kaggle üzerinde yayınlanan BTK Akademi tarafından düzenlenen bir yarışmadan alınmıştır. Verilerin telif ve kullanım koşulları için yarışma sayfasını inceleyiniz.

### Çıktılar
- Eğitim sonrası test seti için oluşturulan tahminler `outputs/lgbm_model_submission.csv` dosyasına yazılır.

