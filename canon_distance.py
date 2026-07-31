# -*- coding: utf-8 -*-
"""
캐논 DSLR 사진에서 물체 크기 기준 거리 측정 (EOS 200D II 검증됨)
==================================================================
크기를 아는 물체(예: 폼롤러 91.3cm)를 찍으면, EXIF 초점거리와
센서 크기(APS-C 22.3mm)로 거리를 계산한다.

실험으로 검증된 정확도 (2.553m 실측 기준):
  18mm: -3.5% / 24mm: -1.1% / 32mm: -0.3%
  → 줌 30mm 이상 권장. 줌별 보정계수 적용 시 전 구간 1% 이내.

주의: 거리는 카메라 윗면 Φ 마크(센서 면)부터 잰다.

사용:
  python canon_distance.py 사진.jpg --height 0.913
  python canon_distance.py 사진.jpg --height 0.913 --true-dist 2.553  # 오차 확인
현재 물체 검출은 보라색 폼롤러 기준(HSV 색 분리). 다른 물체는
--hsv 로 색 범위를 바꾸거나 코드의 detect() 수정.
"""

import argparse
import cv2
import numpy as np
import exifread

SENSOR_LONG_MM = 22.3     # EOS 200D II (APS-C) 긴 변
# 실험 2에서 얻은 줌별 보정계수 (선형 보간)
CAL_TABLE = [(18, 1.036), (24, 1.011), (32, 1.003)]


def cal_factor(f_mm):
    t = CAL_TABLE
    if f_mm <= t[0][0]:
        return t[0][1]
    if f_mm >= t[-1][0]:
        return t[-1][1]
    for (f0, c0), (f1, c1) in zip(t, t[1:]):
        if f0 <= f_mm <= f1:
            r = (f_mm - f0) / (f1 - f0)
            return c0 + r * (c1 - c0)
    return 1.0


def detect(img, hsv_lo=(120, 60, 50), hsv_hi=(158, 255, 255)):
    """길쭉한 단색 물체를 색 분리로 검출 → (긴 축 픽셀, 회전사각형)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lo, hsv_hi)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, bs = None, 0
    for c in cnts:
        if cv2.contourArea(c) < 30000:
            continue
        rect = cv2.minAreaRect(c)
        rw, rh = rect[1]
        if min(rw, rh) == 0:
            continue
        e = max(rw, rh) / min(rw, rh)
        if 3 < e < 12:
            s = cv2.contourArea(c) * e
            if s > bs:
                best, bs = rect, s
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photo")
    ap.add_argument("--height", type=float, required=True, help="물체 실제 길이(m)")
    ap.add_argument("--true-dist", type=float, default=None)
    ap.add_argument("--no-cal", action="store_true", help="줌별 보정 끄기")
    args = ap.parse_args()

    tags = exifread.process_file(open(args.photo, "rb"), details=False)
    f_mm = float(str(tags["EXIF FocalLength"]))
    img = cv2.imdecode(np.fromfile(args.photo, np.uint8), cv2.IMREAD_COLOR)
    long_px = max(img.shape[:2])
    rect = detect(img)
    if rect is None:
        raise SystemExit("물체 검출 실패 — --hsv 범위 조정 필요")
    obj_px = max(rect[1])
    d = f_mm * args.height * 1000 / (obj_px / long_px * SENSOR_LONG_MM) / 1000
    c = 1.0 if args.no_cal else cal_factor(f_mm)
    d_cal = d * c
    print(f"f={f_mm:.0f}mm, 물체 {obj_px:.0f}px")
    print(f"거리: {d_cal:.3f} m (원시 {d:.3f} m x 보정 {c:.3f})")
    if args.true_dist:
        print(f"실측 {args.true_dist:.3f} m → 오차 {(d_cal-args.true_dist)*100:+.1f} cm "
              f"({(d_cal/args.true_dist-1)*100:+.1f}%)")
    box = cv2.boxPoints(rect).astype(int)
    ov = img.copy()
    cv2.drawContours(ov, [box], 0, (0, 255, 0), 8)
    cv2.putText(ov, f"{d_cal:.2f}m", (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 10)
    out = "det_" + args.photo.replace("\\", "/").split("/")[-1]
    ok, buf = cv2.imencode(".jpg", cv2.resize(ov, None, fx=0.25, fy=0.25))
    buf.tofile(out)
    print(f"확인용 이미지 → {out}")


if __name__ == "__main__":
    main()
