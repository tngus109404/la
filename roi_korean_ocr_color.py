# roi_korean_ocr_color.py
import argparse
import cv2
import numpy as np
import sys
import re
from sklearn.cluster import KMeans

# ----- (선택) Tesseract 보조 옵션 -----
USE_TESSERACT_FALLBACK = False
try:
    import pytesseract  # noqa
    USE_TESSERACT_FALLBACK = True
except Exception:
    USE_TESSERACT_FALLBACK = False

# ----- EasyOCR 로더 -----
def load_easyocr():
    try:
        import easyocr
        reader = easyocr.Reader(['ko'], gpu=False)  # 한국어 전용
        return reader
    except Exception as e:
        print("[WARN] EasyOCR 로드 실패:", e)
        return None

# 한글/숫자/기호만 남기기 (영문 억제용)
# 유니코드 한글 범위: ㄱ-ㅎ, ㅏ-ㅣ, 가-힣
KOREAN_KEEP_REGEX = re.compile(r"[가-힣ㄱ-ㅎㅏ-ㅣ0-9\s\.\,\-\+\(\)\[\]<>:;!/·•~_]+")

def korean_only(text: str) -> str:
    # 라인별로 처리해서 완전히 빈 줄은 제거
    lines = []
    for line in text.splitlines():
        kept = "".join(KOREAN_KEEP_REGEX.findall(line))
        kept = re.sub(r"\s+", " ", kept).strip()
        if kept:
            lines.append(kept)
    return "\n".join(lines).strip()

# 간단한 최근접 색 이름 (LAB 거리)
NAMED_COLORS = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (255, 0, 0),
    "lime": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "gray": (128, 128, 128),
    "orange": (255, 165, 0),
    "pink": (255, 192, 203),
    "brown": (165, 42, 42),
    "purple": (128, 0, 128),
    "navy": (0, 0, 128),
    "teal": (0, 128, 128),
    "olive": (128, 128, 0),
}

def rgb_to_hex(rgb):
    r,g,b = [int(max(0, min(255, x))) for x in rgb]
    return f"#{r:02X}{g:02X}{b:02X}"

def nearest_color_name(rgb):
    # RGB -> LAB로 변환 후, LAB 거리로 최근접
    sample = np.uint8([[list(rgb)]])  # shape (1,1,3)
    sample_lab = cv2.cvtColor(sample, cv2.COLOR_RGB2LAB)[0,0].astype(np.float32)

    best_name = None
    best_dist = 1e9
    for name, c_rgb in NAMED_COLORS.items():
        c_lab = cv2.cvtColor(np.uint8([[list(c_rgb)]]), cv2.COLOR_RGB2LAB)[0,0].astype(np.float32)
        d = np.linalg.norm(sample_lab - c_lab)
        if d < best_dist:
            best_dist = d
            best_name = name
    return best_name

def upscale_and_enhance(img):
    # OCR 안정화를 위한 업스케일 + 대비향상
    h, w = img.shape[:2]
    scale = 2 if max(h, w) < 1000 else 1
    if scale > 1:
        img = cv2.resize(img, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
    # LAB에서 L 채널 CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2,a,b])
    img2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    return img2

def ocr_korean_text(roi_bgr, reader):
    # EasyOCR 시도
    text = ""
    if reader is not None:
        result = reader.readtext(roi_bgr, detail=1, paragraph=True)
        # result: list of [bbox, text, confidence]
        lines = []
        for _, txt, conf in result:
            if conf is None or conf < 0:  # 방어적
                continue
            lines.append(txt)
        text = "\n".join(lines).strip()

    # Fallback: Tesseract (kor)
    if (not text) and USE_TESSERACT_FALLBACK:
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        config = "--oem 3 --psm 6 -l kor"
        try:
            raw = pytesseract.image_to_string(roi_rgb, config=config)
            text = raw.strip()
        except Exception:
            pass

    # 한글만 남기기(영문 억제)
    text = korean_only(text)
    return text

def estimate_text_color(roi_bgr):
    """
    아이디어:
    1) Canny로 강한 에지(글자 윤곽) 마스크 추출 → 약간 dilate로 두껍게
    2) 에지 부근 픽셀만 추려서 KMeans(k=3) 클러스터링
    3) 각 클러스터의 평균 '에지 강도'가 높은 쪽을 '글자색'으로 간주
       (윤곽/획을 많이 포함하는 군집이 글자에 해당할 가능성이 높음)
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    # 자동 Canny 임계: median 기반
    med = np.median(gray)
    low = int(max(0, 0.66*med))
    high = int(min(255, 1.33*med))
    edges = cv2.Canny(gray, low, high)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)

    ys, xs = np.where(edges > 0)
    if len(xs) < 50:
        # 에지가 너무 적으면 전체에서 샘플링
        h, w = gray.shape
        ys, xs = np.mgrid[0:h, 0:w].reshape(2, -1)
        sample_pixels = roi_bgr[ys, xs].reshape(-1, 3)
        sample_edges = edges[ys, xs].reshape(-1, 1).astype(np.float32)
    else:
        sample_pixels = roi_bgr[ys, xs].reshape(-1, 3)
        sample_edges = edges[ys, xs].reshape(-1, 1).astype(np.float32)

    # BGR -> RGB로 변환 후 KMeans
    sample_rgb = sample_pixels[:, ::-1].astype(np.float32)  # (N,3)
    # 표본이 너무 많으면 다운샘플
    if sample_rgb.shape[0] > 5000:
        idx = np.random.choice(sample_rgb.shape[0], 5000, replace=False)
        sample_rgb = sample_rgb[idx]
        sample_edges = sample_edges[idx]

    k = 3 if sample_rgb.shape[0] >= 300 else 2
    kmeans = KMeans(n_clusters=k, n_init=8, random_state=42).fit(sample_rgb)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_  # (k,3) in RGB

    # 각 클러스터에 대해 에지 강도 평균 계산 → 높은 쪽을 글자색으로
    best_label, best_score = None, -1.0
    for c in range(k):
        mask = (labels == c)
        if np.sum(mask) == 0:
            continue
        edge_score = float(np.mean(sample_edges[mask]))  # 0~255 범위 평균
        if edge_score > best_score:
            best_score = edge_score
            best_label = c

    text_rgb = centers[best_label] if best_label is not None else np.array([0,0,0], dtype=np.float32)
    color_name = nearest_color_name(text_rgb)
    return text_rgb, rgb_to_hex(text_rgb), color_name, edges

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="스크린샷 이미지 경로")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print("[ERROR] 이미지를 열 수 없습니다:", args.image)
        sys.exit(1)

    disp = img.copy()
    cv2.namedWindow("Drag ROI and press ENTER", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Drag ROI and press ENTER", 1200, 800)
    roi = cv2.selectROI("Drag ROI and press ENTER", disp, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Drag ROI and press ENTER")
    x, y, w, h = map(int, roi)
    if w <= 0 or h <= 0:
        print("[ERROR] ROI가 선택되지 않았습니다.")
        sys.exit(1)

    roi_bgr = img[y:y+h, x:x+w].copy()
    roi_bgr = upscale_and_enhance(roi_bgr)

    reader = load_easyocr()
    text = ocr_korean_text(roi_bgr, reader)

    text_rgb, text_hex, color_name, edges = estimate_text_color(roi_bgr)

    print("\n====== OCR 결과 ======")
    if text:
        print(text)
    else:
        print("(인식된 한글 텍스트 없음)")

    print("\n====== 글자 색 추정 ======")
    r, g, b = [int(round(v)) for v in text_rgb]
    print(f"RGB: ({r}, {g}, {b})  HEX: {text_hex}  이름(근사): {color_name}")

    # 시각화
    vis = roi_bgr.copy()
    # 결과 텍스트를 이미지 위에 살짝 표기
    overlay = vis.copy()
    cv2.rectangle(overlay, (0,0), (vis.shape[1], 60), (0,0,0), -1)
    alpha = 0.6
    vis = cv2.addWeighted(overlay, alpha, vis, 1-alpha, 0)

    label = f"TEXT COLOR ~ RGB({r},{g},{b}) {text_hex} {color_name}"
    cv2.putText(vis, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (b,g,r), 2, cv2.LINE_AA)

    # 에지 맵도 함께 보여주기
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    concat = np.vstack([vis, edges_bgr])

    cv2.namedWindow("ROI result (top) + edges (bottom)", cv2.WINDOW_NORMAL)
    cv2.imshow("ROI result (top) + edges (bottom)", concat)
    print("\n창을 닫으면 프로그램이 종료됩니다.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
