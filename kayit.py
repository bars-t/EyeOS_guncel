"""
EyeOS – Veri Toplama (kayit.py)
─────────────────────────────────────────────────────────────────────
DEĞİŞİKLİKLER:
  • Veriler her çalıştırmada SİLİNMEZ → dosya varsa üstüne eklenir (append).
  • Ekranın sağ üst köşesinde CANLI model doğruluğu gösterilir
    (en az 20 kayıt varsa otomatik modeli eğitip gösterir).
  • Kayıt sayısı ve sınıf dağılımı her an ekranda görünür.

Kontroller:
    0 → Normal bakış
    1 → Sadece SOL göz kapalı
    2 → Sadece SAĞ göz kapalı
    3 → İki göz birlikte kapalı
    q → Çıkış ve kaydet
─────────────────────────────────────────────────────────────────────
"""

import cv2
import mediapipe as mp
import csv
import math
import os
import numpy as np

# ── AYARLAR ──────────────────────────────────────────────────────────
CSV_YOLU = "eyeos_veri.csv"   # Her seferinde aynı dosyaya eklenir
# ─────────────────────────────────────────────────────────────────────

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.7, min_tracking_confidence=0.7
)

LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

ETIKET_ADLARI = {
    0: "Normal",
    1: "Sol Kapali",
    2: "Sag Kapali",
    3: "Ikisi Birden",
}

def calculate_ear(eye_points, landmarks, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_points]
    v1 = math.hypot(pts[1][0] - pts[5][0], pts[1][1] - pts[5][1])
    v2 = math.hypot(pts[2][0] - pts[4][0], pts[2][1] - pts[4][1])
    h_dist = math.hypot(pts[0][0] - pts[3][0], pts[0][1] - pts[3][1])
    return (v1 + v2) / (2.0 * h_dist) if h_dist else 0


def csv_oku():
    """Mevcut CSV'yi okur, (X_listesi, y_listesi) döndürür."""
    rows = []
    if not os.path.exists(CSV_YOLU):
        return [], []
    with open(CSV_YOLU, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append((float(row['Sol_EAR']), float(row['Sag_EAR']), int(row['Etiket'])))
            except Exception:
                pass
    X = [(r[0], r[1]) for r in rows]
    y = [r[2] for r in rows]
    return X, y


def canli_dogruluk():
    """
    Mevcut CSV'deki verilerle anlık çapraz-doğrulama doğruluğunu hesaplar.
    Veri yoksa ya da az ise None döner.
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        import numpy as np

        X, y = csv_oku()
        if len(X) < 20:
            return None, len(X), {}

        X_np = np.array(X)
        y_np = np.array(y)

        # Sınıf dağılımı
        dagilim = {}
        for label in [0, 1, 2, 3]:
            dagilim[label] = int(np.sum(y_np == label))

        # En az 2 sınıf varsa cv yapabilir
        unique = np.unique(y_np)
        if len(unique) < 2:
            return None, len(X), dagilim

        cv = min(3, min(np.bincount(y_np)[np.bincount(y_np) > 0]))
        if cv < 2:
            return None, len(X), dagilim

        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        scores = cross_val_score(clf, X_np, y_np, cv=cv, scoring='accuracy')
        return float(scores.mean()), len(X), dagilim
    except Exception:
        return None, 0, {}


# ── DOSYA HAZIRLA (başlık sadece yoksa yazılır) ──────────────────────
dosya_yeni = not os.path.exists(CSV_YOLU)
csv_dosya = open(CSV_YOLU, mode='a', newline='')
csv_yazici = csv.writer(csv_dosya)
if dosya_yeni:
    csv_yazici.writerow(['Sol_EAR', 'Sag_EAR', 'Etiket'])
    csv_dosya.flush()
print(f"CSV hazır: {os.path.abspath(CSV_YOLU)}")

# ── DURUM DEĞİŞKENLERİ ───────────────────────────────────────────────
dogruluk_cache    = None   # son hesaplanan doğruluk
toplam_kayit      = 0      # bu oturumda eklenen kayıt
guncelle_sayaci   = 0      # her N frame'de bir doğruluğu yenile
GUNCELLE_ARALIGI  = 90     # ~3 sn (30fps)

cap = cv2.VideoCapture(0)

print("\nKAYIT BAŞLADI!")
print("Klavyeden BASILI TUTARAK veri topla:")
for k, v in ETIKET_ADLARI.items():
    print(f"  {k} → {v}")
print("Çıkmak için 'q'.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    key = cv2.waitKey(1) & 0xFF
    etiket = None
    if key == ord('0'): etiket = 0
    elif key == ord('1'): etiket = 1
    elif key == ord('2'): etiket = 2
    elif key == ord('3'): etiket = 3
    elif key == ord('q'): break

    sol_ear = sag_ear = 0.0

    if results.multi_face_landmarks:
        lms = results.multi_face_landmarks[0].landmark
        sol_ear = calculate_ear(LEFT_EYE,  lms, w, h)
        sag_ear = calculate_ear(RIGHT_EYE, lms, w, h)

        # Kayıt
        if etiket is not None:
            csv_yazici.writerow([sol_ear, sag_ear, etiket])
            csv_dosya.flush()
            toplam_kayit += 1
            guncelle_sayaci = 0   # yeni veri → doğruluğu yenile

        # EAR göstergesi
        cv2.putText(frame,
                    f"Sol EAR: {sol_ear:.3f}  |  Sag EAR: {sag_ear:.3f}",
                    (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

        # Kayıt bildirimi
        if etiket is not None:
            etiket_adi = ETIKET_ADLARI.get(etiket, str(etiket))
            cv2.rectangle(frame, (0, 0), (w, 40), (0, 120, 0), -1)
            cv2.putText(frame, f"KAYIT: {etiket_adi}  (+{toplam_kayit} bu oturumda)",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        cv2.putText(frame, "Yuz bulunamadi!", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # ── Canlı doğruluk (her GUNCELLE_ARALIGI frame'de yenile) ────────
    guncelle_sayaci += 1
    if guncelle_sayaci >= GUNCELLE_ARALIGI:
        guncelle_sayaci = 0
        dogruluk_cache = canli_dogruluk()

    # Sağ üst panelde doğruluk ve istatistik
    panel_x = w - 280
    cv2.rectangle(frame, (panel_x - 5, 0), (w, 200), (30, 30, 30), -1)

    if dogruluk_cache is None:
        cv2.putText(frame, "Hesaplaniyor...", (panel_x, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    else:
        acc, n_toplam, dagilim = dogruluk_cache
        acc_str = f"%{acc*100:.1f}" if acc is not None else "Yetersiz veri"
        renk = (0, 255, 100) if (acc or 0) > 0.85 else (0, 200, 255)

        cv2.putText(frame, "MODEL DOGRULUGU", (panel_x, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, acc_str, (panel_x, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, renk, 2)
        cv2.putText(frame, f"Toplam: {n_toplam} kayit", (panel_x, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

        # Sınıf dağılımı
        for i, (lbl, cnt) in enumerate(dagilim.items()):
            cv2.putText(frame,
                        f"  [{lbl}] {ETIKET_ADLARI[lbl]}: {cnt}",
                        (panel_x, 110 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

    # Başlık
    cv2.rectangle(frame, (0, 0), (panel_x - 5, 28), (25, 25, 25), -1)
    cv2.putText(frame, "EyeOS - Veri Toplama", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("EyeOS - Veri Toplama", frame)

csv_dosya.close()
cap.release()
cv2.destroyAllWindows()
print(f"\nBitti! Bu oturumda {toplam_kayit} kayıt eklendi.")
print(f"Toplam dosya: {os.path.abspath(CSV_YOLU)}")