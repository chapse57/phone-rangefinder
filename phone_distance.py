# -*- coding: utf-8 -*-
"""
스마트폰 사진 기반 거리 측정 — 연구식 반복 개선 파이프라인
=============================================================
워크플로우:
  1. python phone_distance.py marker
       → 인쇄용 A4 마커 시트 생성 (marker_print_A4.png, 실제크기 100%로 인쇄)
  2. 폰으로 마커 사진을 여러 장 촬영 → photos/ 폴더에 복사
  3. python phone_distance.py measure --photos photos
       → 거리 추정 (반복 측정 + IQR 이상치 제거 + 평균)
  4. (정확도 개선) 줄자로 잰 실제 거리를 알려주고 캘리브레이션:
     python phone_distance.py calibrate --photos photos --true-dist 1.00
       → 카메라 초점거리를 역산해 camera.json 저장, 이후 measure가 자동 사용
  5. measure에 --true-dist를 주면 오차가 experiment_log.csv에 기록되어
     실험을 거듭할수록 정확도가 얼마나 개선되는지 추적할 수 있음

초점거리(카메라 기종 차이) 결정 우선순위:
  camera.json(캘리브레이션 값) > 사진 EXIF(35mm 환산 초점거리) > 기본 FOV 가정
촬영 각도는 solvePnP 자세 추정이 자동 보정 (기울여 찍어도 됨).
마커 크기는 기본 10 cm 강제, 다른 크기 사용 시 --marker-size 로 지정.

필요 패키지: pip install opencv-contrib-python pillow numpy
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime

import numpy as np
import cv2

# ------------------------- 고정 설정 (한 가지로 강제) -------------------------
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
MARKER_ID = 7
DEFAULT_MARKER_SIZE = 0.10        # m (인쇄 시트와 반드시 일치해야 함)
DEFAULT_HFOV_DEG = 67.0           # EXIF도 캘리브레이션도 없을 때의 가정값
CAMERA_JSON = "camera.json"
LOG_CSV = "experiment_log.csv"

_params = cv2.aruco.DetectorParameters()
_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT, _params)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def imread_unicode(path):
    """한글 경로에서도 동작하는 이미지 로드 (Windows cv2.imread 한계 우회)."""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


# ------------------------- 초점거리 결정 -------------------------
def exif_focal_35mm(path):
    """EXIF에서 35mm 환산 초점거리를 읽음. 없으면 None."""
    try:
        from PIL import Image, ExifTags
        img = Image.open(path)
        exif = img.getexif()
        if not exif:
            return None
        ifd = exif.get_ifd(ExifTags.IFD.Exif)
        f35 = ifd.get(ExifTags.Base.FocalLengthIn35mmFilm)
        return float(f35) if f35 else None
    except Exception:
        return None


def get_intrinsics(img_w, img_h, photo_path=None, verbose=False):
    """(K, 출처설명) 반환. 우선순위: camera.json > EXIF > FOV 가정."""
    if os.path.exists(CAMERA_JSON):
        with open(CAMERA_JSON, encoding="utf-8") as f:
            cam = json.load(f)
        # 캘리브레이션 당시 해상도와 다르면 비례 스케일
        scale = img_w / cam["img_w"]
        fx = cam["fx"] * scale
        src = f"camera.json (캘리브레이션, fx={fx:.0f}px)"
    else:
        f35 = exif_focal_35mm(photo_path) if photo_path else None
        if f35:
            # 35mm 필름 가로 36mm 기준 환산
            long_side = max(img_w, img_h)
            fx = long_side * f35 / 36.0
            src = f"EXIF 35mm환산 {f35:.0f}mm (fx={fx:.0f}px)"
        else:
            fx = (img_w / 2) / np.tan(np.deg2rad(DEFAULT_HFOV_DEG / 2))
            src = f"기본 FOV {DEFAULT_HFOV_DEG}° 가정 (fx={fx:.0f}px, 부정확할 수 있음)"
    K = np.array([[fx, 0, img_w / 2],
                  [0, fx, img_h / 2],
                  [0,  0, 1]], dtype=np.float64)
    if verbose:
        print(f"   초점거리 출처: {src}")
    return K, src


# ------------------------- 1회 측정 -------------------------
def obj_pts(size):
    h = size / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]],
                    dtype=np.float64)


def measure_photo(path, marker_size, verbose=True):
    """사진 1장 → (거리 m, 기울기 deg, 초점거리 출처) 또는 None."""
    img = imread_unicode(path)
    if img is None:
        if verbose:
            print(f" ! {os.path.basename(path)}: 이미지를 열 수 없음")
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = DETECTOR.detectMarkers(gray)
    if ids is None or MARKER_ID not in ids.flatten():
        if verbose:
            print(f" ! {os.path.basename(path)}: 마커(ID {MARKER_ID}) 미검출")
        return None
    idx = list(ids.flatten()).index(MARKER_ID)
    pts = corners[idx].reshape(-1, 2).astype(np.float64)

    K, src = get_intrinsics(img.shape[1], img.shape[0], path)
    ok, rvec, tvec = cv2.solvePnP(obj_pts(marker_size), pts, K, np.zeros(5),
                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    dist = float(np.linalg.norm(tvec))
    R, _ = cv2.Rodrigues(rvec)
    tilt = float(np.rad2deg(np.arccos(np.clip(abs(R[2, 2]), -1, 1))))
    if verbose:
        print(f" - {os.path.basename(path)}: {dist:.3f} m (기울기 {tilt:.0f}°)")
    return dist, tilt, src


# ------------------------- 반복 측정 통합 -------------------------
def robust_estimate(samples):
    """IQR 이상치 제거 + 평균. (추정값, 사용표본수, 표준편차) 반환."""
    s = np.asarray(samples, dtype=np.float64)
    if len(s) == 0:
        return None, 0, None
    if len(s) < 4:
        return float(np.median(s)), len(s), float(np.std(s))
    q1, q3 = np.percentile(s, [25, 75])
    iqr = q3 - q1
    keep = s[(s >= q1 - 1.5 * iqr) & (s <= q3 + 1.5 * iqr)]
    if len(keep) == 0:
        keep = s
    return float(keep.mean()), len(keep), float(keep.std())


def collect_photos(folder):
    if not os.path.isdir(folder):
        sys.exit(f"폴더가 없습니다: {folder}\n폰 사진을 이 폴더에 복사해 주세요.")
    files = sorted(f for f in os.listdir(folder)
                   if f.lower().endswith(IMG_EXTS))
    if not files:
        sys.exit(f"{folder} 안에 사진이 없습니다 (jpg/png). "
                 "아이폰 HEIC는 JPG로 변환 후 넣어주세요.")
    return [os.path.join(folder, f) for f in files]


# ------------------------- 명령: measure -------------------------
def cmd_measure(args):
    paths = collect_photos(args.photos)
    print(f"사진 {len(paths)}장 분석 (마커 크기 {args.marker_size*100:.0f} cm)")
    results, src = [], ""
    for p in paths:
        r = measure_photo(p, args.marker_size)
        if r:
            results.append(r)
            src = r[2]
    dists = [r[0] for r in results]
    est, n_used, std = robust_estimate(dists)
    if est is None:
        sys.exit("측정 실패: 마커가 검출된 사진이 없습니다.")

    print("\n================ 결과 ================")
    print(f" 추정 거리   : {est:.3f} m")
    print(f" 사용 표본   : {n_used}/{len(paths)}장 (이상치 {len(dists)-n_used}장 제외)")
    print(f" 표준편차    : {std*1000:.1f} mm")
    print(f" 초점거리    : {src}")
    if args.true_dist:
        err = (est - args.true_dist) * 1000
        print(f" 실제 거리   : {args.true_dist:.3f} m → 오차 {err:+.1f} mm "
              f"({abs(err)/args.true_dist/10:.2f}%)")
        log_session(args.true_dist, est, err, n_used, std, src)
        print(f" 실험 일지에 기록됨 → {LOG_CSV}")
    else:
        print(" (실제 거리를 알면 --true-dist 1.00 처럼 지정 → 오차가 일지에 기록됨)")
    if "FOV" in src:
        print("\n [정확도 개선 팁] 캘리브레이션을 하면 크게 정확해집니다:")
        print("   python phone_distance.py calibrate --photos photos --true-dist <실측거리>")


def log_session(true_d, est, err_mm, n, std, src):
    new = not os.path.exists(LOG_CSV)
    with open(LOG_CSV, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["시각", "실제거리_m", "추정거리_m", "오차_mm",
                        "표본수", "표준편차_mm", "초점거리_출처"])
        w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M"),
                    f"{true_d:.3f}", f"{est:.4f}", f"{err_mm:+.1f}",
                    n, f"{std*1000:.1f}", src])


# ------------------------- 명령: calibrate -------------------------
def cmd_calibrate(args):
    """알려진 거리에서 찍은 사진들로 초점거리 fx를 역산."""
    paths = collect_photos(args.photos)
    print(f"캘리브레이션: 실제 거리 {args.true_dist:.3f} m, 사진 {len(paths)}장")
    fxs, wh = [], None
    for p in paths:
        img = imread_unicode(p)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = DETECTOR.detectMarkers(gray)
        if ids is None or MARKER_ID not in ids.flatten():
            print(f" ! {os.path.basename(p)}: 마커 미검출, 제외")
            continue
        idx = list(ids.flatten()).index(MARKER_ID)
        pts = corners[idx].reshape(-1, 2).astype(np.float64)
        h_img, w_img = img.shape[:2]
        wh = (w_img, h_img)

        # fx를 이분탐색: solvePnP 거리가 실측 거리와 같아지는 fx를 찾음
        lo, hi = 200.0, 20000.0
        for _ in range(60):
            mid = (lo + hi) / 2
            K = np.array([[mid, 0, w_img/2], [0, mid, h_img/2], [0, 0, 1]])
            ok, _, tvec = cv2.solvePnP(obj_pts(args.marker_size), pts, K,
                                       np.zeros(5),
                                       flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok:
                break
            d = float(np.linalg.norm(tvec))
            if d < args.true_dist:
                lo = mid       # 거리가 짧게 나오면 fx를 키움
            else:
                hi = mid
        else:
            fx = (lo + hi) / 2
            fxs.append(fx)
            print(f" - {os.path.basename(p)}: fx = {fx:.0f} px")

    est_fx, n, std = robust_estimate(fxs)
    if est_fx is None:
        sys.exit("캘리브레이션 실패: 유효한 사진이 없습니다.")
    with open(CAMERA_JSON, "w", encoding="utf-8") as f:
        json.dump({"fx": est_fx, "img_w": wh[0], "img_h": wh[1],
                   "calibrated_at": datetime.now().isoformat(),
                   "true_dist_m": args.true_dist, "n_photos": n},
                  f, ensure_ascii=False, indent=2)
    print(f"\n완료: fx = {est_fx:.0f} px (표본 {n}장, 편차 {std:.0f}px)")
    print(f"저장됨 → {CAMERA_JSON}  (이후 measure가 자동으로 사용)")


# ------------------------- 명령: marker -------------------------
def cmd_marker(args):
    """A4 300DPI 인쇄용 마커 시트. 마커 한 변 = 10 cm."""
    dpi = 300
    a4_w, a4_h = int(8.27 * dpi), int(11.69 * dpi)      # 2481 x 3507
    mm2px = dpi / 25.4
    marker_px = int(100 * mm2px)                          # 100 mm
    sheet = np.full((a4_h, a4_w), 255, dtype=np.uint8)
    m = cv2.aruco.generateImageMarker(ARUCO_DICT, MARKER_ID, marker_px)
    x0 = (a4_w - marker_px) // 2
    y0 = int(30 * mm2px)
    sheet[y0:y0+marker_px, x0:x0+marker_px] = m

    # 검증용 100mm 눈금자
    ry = y0 + marker_px + int(20 * mm2px)
    rx = x0
    cv2.line(sheet, (rx, ry), (rx + marker_px, ry), 0, 4)
    for i in range(11):
        x = rx + int(i * 10 * mm2px)
        cv2.line(sheet, (x, ry - int(3*mm2px)), (x, ry + int(3*mm2px)), 0, 4)
    cv2.putText(sheet, "100 mm check ruler  (print at 100% / actual size)",
                (rx, ry + int(10 * mm2px)), cv2.FONT_HERSHEY_SIMPLEX,
                1.5, 0, 3)
    cv2.putText(sheet, f"ArUco DICT_5X5_50  ID {MARKER_ID}   marker = 100 mm",
                (x0, y0 - int(8 * mm2px)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 0, 3)
    ok, buf = cv2.imencode(".png", sheet)
    buf.tofile(args.out)
    print(f"저장됨 → {args.out}")
    print("반드시 '실제 크기(100%)'로 인쇄하고, 눈금자가 100mm인지 자로 확인하세요.")


# ------------------------- main -------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("measure", help="사진 폴더 분석해 거리 추정")
    p1.add_argument("--photos", default="photos")
    p1.add_argument("--marker-size", type=float, default=DEFAULT_MARKER_SIZE,
                    help="마커 한 변(m), 기본 0.10")
    p1.add_argument("--true-dist", type=float, default=None,
                    help="줄자로 잰 실제 거리(m) — 지정 시 오차를 일지에 기록")
    p1.set_defaults(func=cmd_measure)

    p2 = sub.add_parser("calibrate", help="알려진 거리 사진들로 초점거리 보정")
    p2.add_argument("--photos", default="photos")
    p2.add_argument("--true-dist", type=float, required=True)
    p2.add_argument("--marker-size", type=float, default=DEFAULT_MARKER_SIZE)
    p2.set_defaults(func=cmd_calibrate)

    p3 = sub.add_parser("marker", help="A4 인쇄용 마커 시트 생성")
    p3.add_argument("--out", default="marker_print_A4.png")
    p3.set_defaults(func=cmd_marker)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
