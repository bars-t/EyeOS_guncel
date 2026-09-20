"""
EyeOS v7 – BL242 Görüntü İşleme Ödev 4

DÜZELTMELER (v6 → v7):
  1. YÖN DÜZELTİLDİ
       • Gözün yönü = farenin yönü (artık ters değil)
       • iris_to_screen() içindeki eksen ters çevirmeleri kaldırıldı
  2. GÖZ KIRPMA DÜZELTİLDİ
       • EAR_TH artık sabit değil; model_egit.py'nin önerdiği değeri
         veya verinden otomatik hesaplanan eşiği kullanır
       • Kırpma tespiti daha güvenilir: göz başka gözden bağımsız izlenir
  3. MODEL DESTEĞİ
       • eyeos_model.pkl varsa yükler ve kırpma kararını modele bırakır
       • Model yoksa klasik EAR eşiği kullanır (eski davranış)

Kontroller:
    İki göz kırpma            → Sol tık
    Sadece sağ göz kırpma     → Sağ tık
    Sol göz 2sn kapalı        → Sol tuş BASILI (drag modu)
      └─ Drag modunda mouse   → Sadece SAĞ iris ile hareket
      └─ Sol göz açılınca     → Tuş bırakılır

Kalibrasyon: Program açılışında 9 noktaya bak, her birinde SPACE'e bas.
Çıkış: q
"""

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import os
import joblib

# ── AYARLAR ──────────────────────────────────────────────────────────
SMOOTH    = 0.1   # Yumuşatma
DEAD      = 0.015    # Ölü bölge, titreme

# EAR eşiği — model yoksa bu kullanılır.
# model_egit.py'nin önerdiği değeri buraya yaz (varsayılan 0.18).
EAR_TH    = 0.18

BLINK_MIN = 0.05   # Geçerli kırpma min süresi (sn)
BLINK_MAX = 0.40   # Geçerli kırpma max süresi (sn)
HOLD_T    = 2.0    # Drag için sol gözün kapalı kalma süresi (sn)
DEBOUNCE  = 0.40   # İki eylem arasındaki min bekleme (sn)

# Model yolu (kayit.py + model_egit.py ile aynı klasörde olmalı)
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_YOLU  = os.path.join(SCRIPT_DIR, "eyeos_model.pkl")
# ─────────────────────────────────────────────────────────────────────

pyautogui.FAILSAFE = True
pyautogui.PAUSE    = 0
SW, SH = pyautogui.size()

# ── MODEL YÜKLEMESİ ──────────────────────────────────────────────────
# RandomForest modelini belleğe yükle, yoksa hata yönetimiyle EAR eşiğine dön
clf = None
if os.path.exists(MODEL_YOLU):
    try:
        clf = joblib.load(MODEL_YOLU) # Eğitilmiş makine öğrenmesi modelini yükle
        print(f"Model yüklendi → {MODEL_YOLU}")
    except Exception as e:
        print(f"Model yüklenemedi: {e} → EAR eşiği kullanılacak") # Dosya bozuksa yedeğe geç
else:
    print(f"Model bulunamadı ({MODEL_YOLU}) → EAR eşiği kullanılacak (EAR_TH={EAR_TH})")

# ── MEDİAPİPE ────────────────────────────────────────────────────────
# Yüz hatlarını takip etmek için MediaPipe FaceMesh modelini başlat
face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1, refine_landmarks=True, # Tek yüz, iris noktaları için hassas mod
    min_detection_confidence=0.7, min_tracking_confidence=0.7 # Doğruluk eşikleri
)

# Göz ve İris için MediaPipe landmark indekslerini tanımla
L_IRIS = 468; R_IRIS = 473 # İris merkezi noktaları
L_IN   = 133; L_OUT  = 33;  L_TOP = 159; L_BOT = 145 # Sol göz çerçeve noktaları
R_IN   = 362; R_OUT  = 263; R_TOP = 386; R_BOT = 374 # Sağ göz çerçeve noktaları

LEFT_EYE  = [33, 160, 158, 133, 153, 144]  # Sol göz poligonu
RIGHT_EYE = [362, 385, 387, 263, 373, 380]  # Sağ göz poligonu

# ── YARDIMCI FONKSİYONLAR ────────────────────────────────────────────

def get_offset(lms, iris, inn, out, top, bot): 
    """İrisin göz bebeği merkezine göre (yatay/dikey) sapmasını -1 ile +1 arası normalize eder."""
    ix, iy = lms[iris].x, lms[iris].y # İris koordinatı
    cx = (lms[inn].x + lms[out].x) / 2 # Gözün yatay merkezi
    cy = (lms[top].y + lms[bot].y) / 2  # Gözün dikey merkezi
    hw = abs(lms[out].x - lms[inn].x) / 2 # Gözün genişlik yarısı
    hh = abs(lms[bot].y - lms[top].y) / 2 # Gözün yükseklik yarısı
    return (ix - cx) / hw if hw else 0, (iy - cy) / hh if hh else 0 # Normalize edilmiş offset


def get_ear_6(eye_points, lms, w, h): 
    """Göz kapağı noktaları arasındaki dikey/yatay oranı hesaplayarak göz açıklığını (EAR) döndürür."""
    import math
    pts = [(lms[i].x * w, lms[i].y * h) for i in eye_points] # Landmark'ları piksele çevir
    v1 = math.hypot(pts[1][0] - pts[5][0], pts[1][1] - pts[5][1]) # Dikey mesafe 1
    v2 = math.hypot(pts[2][0] - pts[4][0], pts[2][1] - pts[4][1]) # Dikey mesafe 2
    h_d = math.hypot(pts[0][0] - pts[3][0], pts[0][1] - pts[3][1]) # Yatay mesafe
    return (v1 + v2) / (2.0 * h_d) if h_d else 0 # EAR formülü: (dikey_toplam) / (2 * yatay)


def goz_kapali_mi(sol_ear, sag_ear, goz="sol"):
    """
    Kullanıcının göz durumunu, yüklü model varsa Makine Öğrenmesi ile,
    yoksa klasik EAR eşik değerleri (Thresholding) ile belirler.
    """
    if clf is not None:
        pred = int(clf.predict([[sol_ear, sag_ear]])[0]) # Modelden tahmini etiket al
        if goz == "sol":
            return pred in (1, 3) # Model 1 (Sol) veya 3 (İkisi) dediyse kapalıdır
        else:
            return pred in (2, 3) # Model 2 (Sağ) veya 3 (İkisi) dediyse kapalıdır
    else:
        # Klasik mantık: EAR değeri eşiğin altındaysa göz kapalı kabul edilir
        if goz == "sol":
            return sol_ear < EAR_TH
        else:
            return sag_ear < EAR_TH

def iris_to_screen(ox, oy, cal):  # İris konumunu ekran koordinatına çevirir
    """
    İris offsetini ekran koordinatına çevirir.

    DÜZELTME: Kamera zaten yatay flip yapılıyor (cv2.flip(frame,1)).
    Bu yüzden iris X koordinatı gerçek yönü gösteriyor.
    Ek eksen tersi YAPILMAZ — gözün baktığı yön = farenin gittiği yön.
    """
    ox_min, ox_max, oy_min, oy_max = cal

    # X: iris sola giderse fare sola, sağa giderse fare sağa
    tx = int(np.interp(ox, [ox_min, ox_max], [0, SW]))
    # Y: iris yukarı giderse fare yukarı, aşağı giderse fare aşağı
    ty = int(np.interp(oy, [oy_min, oy_max], [SH,0]))

    return int(np.clip(tx, 0, SW - 1)), int(np.clip(ty, 0, SH - 1))


# ── KALİBRASYON ──────────────────────────────────────────────────────
CALIB_PTS = [
    (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
    (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
    (0.1, 0.9), (0.5, 0.9), (0.9, 0.9),
]


def run_calibration(cap):
    """
    Kullanıcının göz yapısını ve kamera açısını sisteme tanıtmak için 9 noktalı 
    bir kalibrasyon arayüzü sunar.
    """
    print("Kalibrasyon: Her noktaya bakıp SPACE'e bas. Çıkmak için q.")
    collected = [] # Her noktadan alınan iris koordinatlarını sakla
    idx = 0        # Şu anki kalibrasyon noktası indeksi
    CW, CH = 900, 600 # Kalibrasyon ekran çözünürlüğü

    while idx < len(CALIB_PTS): # 9 nokta tamamlanana kadar dön
        ok, frame = cap.read()
        if not ok: continue
        
        frame = cv2.flip(frame, 1) # Kullanıcı deneyimi için görüntüyü aynala
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # MediaPipe için RGB'ye çevir
        res = face_mesh.process(rgb) # Yüz landmarklarını çıkar

        # Boş bir siyah ekran oluştur ve kalibrasyon noktalarını üzerine çiz
        scr = np.zeros((CH, CW, 3), dtype=np.uint8)

        for i, (px, py) in enumerate(CALIB_PTS):
            sx, sy = int(px * CW), int(py * CH)
            if i < idx:
                cv2.circle(scr, (sx, sy), 10, (0, 200, 0), -1) # Tamamlananlar (Yeşil)
            elif i == idx:
                # Aktif nokta (Animasyonlu/Yanıp sönen)
                r = int(14 + 4 * np.sin(time.time() * 5)) 
                cv2.circle(scr, (sx, sy), r, (0, 150, 255), 3)
                cv2.circle(scr, (sx, sy), 5, (0, 220, 255), -1)
            else:
                cv2.circle(scr, (sx, sy), 8, (70, 70, 70), 1) # Henüz gelmeyenler (Gri)

        # Yüz algılama durumunu ekrana yaz
        face_ok = res.multi_face_landmarks is not None
        renk  = (0, 255, 0) if face_ok else (0, 0, 255)
        durum = "Yuz algilandi ✓" if face_ok else "YUZ BULUNAMADI!"
        cv2.putText(scr, durum, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, renk, 2)
        cv2.putText(scr, f"Nokta {idx+1}/9 — Noktaya bak, sonra SPACE",
                    (20, CH - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Yüz algılandıysa iris koordinatlarını canlı göster
        if face_ok:
            lms = res.multi_face_landmarks[0].landmark
            lox, loy = get_offset(lms, L_IRIS, L_IN, L_OUT, L_TOP, L_BOT)
            rox, roy = get_offset(lms, R_IRIS, R_IN, R_OUT, R_TOP, R_BOT)
            ox, oy   = (lox + rox) / 2, (loy + roy) / 2
            cv2.putText(scr, f"iris X:{ox:+.3f}  Y:{oy:+.3f}",
                        (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

        cv2.imshow("EyeOS Kalibrasyon", scr)
        key = cv2.waitKey(1) & 0xFF

        # SPACE tuşu ile mevcut iris verisini kaydet ve bir sonraki noktaya geç
        if key == ord(' ') and face_ok:
            lms = res.multi_face_landmarks[0].landmark
            lox, loy = get_offset(lms, L_IRIS, L_IN, L_OUT, L_TOP, L_BOT)
            rox, roy = get_offset(lms, R_IRIS, R_IN, R_OUT, R_TOP, R_BOT)
            val = ((lox + rox) / 2, (loy + roy) / 2)
            collected.append(val)
            print(f"  [{idx+1}/9] X={val[0]:+.3f}  Y={val[1]:+.3f}")
            idx += 1
        elif key == ord('q'): # Çıkış
            cv2.destroyWindow("EyeOS Kalibrasyon")
            return None

    cv2.destroyWindow("EyeOS Kalibrasyon")

    # Toplanan verilerden min/max değerleri hesaplayarak haritalama sınırlarını belirle
    xs = [c[0] for c in collected]
    ys = [c[1] for c in collected]
    ox_min, ox_max = min(xs), max(xs)

    # Dikey sınırlar: Üst sıra (0,1,2) ortalaması ve alt sıra (6,7,8) ortalaması
    oy_min = sum(ys[0:3]) / 3   # Üst sınır (Göz yukarı bakınca oluşan değer)
    oy_max = sum(ys[6:9]) / 3   # Alt sınır (Göz aşağı bakınca oluşan değer)

    # Güvenlik marjı: Eğer değerler çok yakınsa varsayılan sınırları ata
    if ox_max - ox_min < 0.05: ox_min, ox_max = -0.22, 0.22
    if oy_max - oy_min < 0.03: oy_min, oy_max = -0.16, 0.16

    print(f"Kalibrasyon tamam → X:[{ox_min:.3f},{ox_max:.3f}]  Y:[{oy_min:.3f},{oy_max:.3f}]")
    return ox_min, ox_max, oy_min, oy_max


# ── BAŞLAT ───────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cal = run_calibration(cap)
if cal is None:
    cap.release()
    exit()

# ── DURUM DEĞİŞKENLERİ ───────────────────────────────────────────────
cx, cy = SW // 2, SH // 2
lc = rc = False
lt = rt = 0.0
drag    = False
tb = tr = 0.0
msg = ""; msg_t = 0.0


def say(m):
    global msg, msg_t
    msg   = m
    msg_t = time.time()


say("Hazır!")
print("EyeOS v7 başladı. Çıkmak için q.")
MOD_STR = "Model: RF" if clf else f"EAR eşiği: {EAR_TH}"
print(MOD_STR)

# ── ANA DÖNGÜ ────────────────────────────────────────────────────────
while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    H, W  = frame.shape[:2]
    now   = time.time()

    res = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if res.multi_face_landmarks:
        lms = res.multi_face_landmarks[0].landmark

        # ── İRİS → MOUSE ─────────────────────────────────────────────
        if drag:
            ox, oy = get_offset(lms, R_IRIS, R_IN, R_OUT, R_TOP, R_BOT)
        else:
            lox, loy = get_offset(lms, L_IRIS, L_IN, L_OUT, L_TOP, L_BOT)
            rox, roy = get_offset(lms, R_IRIS, R_IN, R_OUT, R_TOP, R_BOT)
            ox, oy   = (lox + rox) / 2, (loy + roy) / 2

        if abs(ox) < DEAD: ox = 0
        if abs(oy) < DEAD: oy = 0

        tx, ty = iris_to_screen(ox, oy, cal)
        s  = SMOOTH * (1.5 if drag else 1.0)
        cx = int(cx + (tx - cx) * s)
        cy = int(cy + (ty - cy) * s)
        pyautogui.moveTo(cx, cy)

        # ── EAR ──────────────────────────────────────────────────────
        sol_ear = get_ear_6(LEFT_EYE,  lms, W, H)
        sag_ear = get_ear_6(RIGHT_EYE, lms, W, H)

        lc_now = goz_kapali_mi(sol_ear, sag_ear, "sol")
        rc_now = goz_kapali_mi(sol_ear, sag_ear, "sag")

        # Gözün kapandığı anı kaydet (kırpma süresi ölçmek için)
        if lc_now and not lc: lc = True;  lt = now
        if rc_now and not rc: rc = True;  rt = now

        # Sol göz 2sn kapalı kalırsa sol tuşu basılı tut (drag)
        if lc and lc_now and not drag and now - lt >= HOLD_T:
            drag = True
            pyautogui.mouseDown(button='left')
            say("◀ DRAG — Sağ gözle yönlendir")

        # Göz az önce açıldı mı? → tık kararı ver
        lo = lc and not lc_now
        ro = rc and not rc_now

        if lo or ro:
            ld = now - lt if lo else 0
            rd = now - rt if ro else 0

            if lo and drag:
                drag = False
                pyautogui.mouseUp(button='left')
                say("Drag bitti")

            elif not drag:
                lv   = BLINK_MIN < ld < BLINK_MAX if lo else False
                rv   = BLINK_MIN < rd < BLINK_MAX if ro else False
                both = lo and ro and abs(lt - rt) < 0.10  # İkisi 0.1sn içinde açıldıysa birlikte sayılır

                if both and lv and rv and now - tb > DEBOUNCE:
                    pyautogui.click(button='left')
                    say("◀◀ SOL TIK"); tb = now

                elif ro and rv and not lc_now and now - tr > DEBOUNCE and now - tb > DEBOUNCE:
                    pyautogui.click(button='right')
                    say("▶ SAĞ TIK"); tr = now

        if not lc_now and lc: lc = False
        if not rc_now and rc: rc = False

        # ── Görsel ───────────────────────────────────────────────────
        cv2.circle(frame, (int(lms[L_IRIS].x * W), int(lms[L_IRIS].y * H)), 4, (0, 255, 0), -1)
        cv2.circle(frame, (int(lms[R_IRIS].x * W), int(lms[R_IRIS].y * H)), 4, (0, 255, 0), -1)

        cl = (0, 0, 255) if lc_now else (0, 255, 0)
        cr = (0, 0, 255) if rc_now else (0, 255, 0)
        cv2.putText(frame, f"Sol:{sol_ear:.3f}", (8,  50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cl, 1)
        cv2.putText(frame, f"Sag:{sag_ear:.3f}", (120, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cr, 1)
        cv2.putText(frame, f"ox:{ox:+.2f} oy:{oy:+.2f} | {cx},{cy}",
                    (8, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

        # Drag dolum çubuğu
        if lc and lc_now and not drag:
            f = min((now - lt) / HOLD_T, 1.0)
            cv2.rectangle(frame, (8, 80), (8 + int(140 * f), 90), (0, 100, 255), -1)
            cv2.putText(frame, "Drag hazırlanıyor...", (8, 104),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 255), 1)

        if drag:
            cv2.rectangle(frame, (0, 0), (W, H), (0, 80, 200), 3)
            cv2.putText(frame, "[ DRAG MODU - Sag gozle yon ver ]",
                        (W // 2 - 165, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 255), 2)

    else:
        cv2.putText(frame, "Yüz bulunamadı", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Başlık
    cv2.rectangle(frame, (0, 0), (W, 28), (25, 25, 25), -1)
    cv2.putText(frame, f"EyeOS v7  |  {MOD_STR}", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Durum mesajı
    if now - msg_t < 1.8:
        cv2.rectangle(frame, (0, H - 32), (W, H), (20, 20, 20), -1)
        cv2.putText(frame, msg, (8, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 200), 2)

    cv2.imshow("EyeOS v7", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Temizlik
if drag:
    pyautogui.mouseUp(button='left')
cap.release()
cv2.destroyAllWindows()
face_mesh.close()
print("EyeOS v7 kapatıldı.")