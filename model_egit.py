"""
EyeOS – Model Eğitimi
━━━━━━━━━━━━━━━━━━━━
kayit.py ile toplanan veriyi okur, RandomForest modeli eğitir,
doğruluk raporunu gösterir ve modeli kaydeder.

Kullanım: python model_egit.py
"""

import os, pandas as pd, numpy as np, joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
CSV_YOLU      = os.path.join(SCRIPT_DIR, "eyeos_veri.csv")
MODEL_CIKTISI = os.path.join(SCRIPT_DIR, "eyeos_model.pkl")

ETIKET = {0:"Normal", 1:"Sol Kapali", 2:"Sag Kapali", 3:"Ikisi Birden"}

if not os.path.exists(CSV_YOLU):
    print(f"HATA: CSV bulunamadı → {CSV_YOLU}")
    exit()

df = pd.read_csv(CSV_YOLU)
print(f"\nToplam kayıt: {len(df)}")
print("Sınıf dağılımı:")
for lbl, cnt in df['Etiket'].value_counts().sort_index().items():
    print(f"  [{lbl}] {ETIKET.get(lbl,'?')}: {cnt} kayıt")

X = df[['Sol_EAR', 'Sag_EAR']].values
y = df['Etiket'].values

# EAR istatistikleri + EAR_TH önerisi
print("\nEAR İstatistikleri:")
for lbl in sorted(np.unique(y)):
    mask = y == lbl
    print(f"  [{lbl}] {ETIKET.get(lbl,'?')}: "
          f"Sol={X[mask,0].mean():.3f}±{X[mask,0].std():.3f}  "
          f"Sag={X[mask,1].mean():.3f}±{X[mask,1].std():.3f}")

normal = y == 0
if normal.sum() > 0:
    onerilen = round((X[normal,0].mean() - 2*X[normal,0].std() +
                      X[normal,1].mean() - 2*X[normal,1].std()) / 2, 3)
    print(f"\n  ★ Önerilen EAR_TH = {onerilen}  (eyeos_v2.py içinde EAR_TH değeri)")

# Eğit
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc    = accuracy_score(y_test, y_pred)

cv = min(5, int(np.bincount(y.astype(int)).min()))
cv_scores = cross_val_score(model, X, y, cv=max(2,cv), scoring='accuracy')

print(f"\nDoğruluk (test)      : %{acc*100:.1f}")
print(f"Çapraz doğrulama     : %{cv_scores.mean()*100:.1f} ± {cv_scores.std()*100:.1f}")
print("\nDetaylı rapor:")
print(classification_report(y_test, y_pred,
      target_names=[ETIKET.get(i,str(i)) for i in sorted(np.unique(y))]))
print("Karışıklık matrisi:")
print(confusion_matrix(y_test, y_pred))

joblib.dump(model, MODEL_CIKTISI)
print(f"\nModel kaydedildi → {MODEL_CIKTISI}")