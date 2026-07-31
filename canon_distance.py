# -*- coding: utf-8 -*-
"""
캐논 DSLR 사진에서 물체 크기 기준 거리 측정 (EOS 200D II 검증됨)
==================================================================
크기를 아는 물체를 찍으면 EXIF 초점거리와 센서 크기(APS-C 22.3mm)로
거리를 계산한다. 거리는 카메라 윗면 Φ 마크(센서 면)부터 잰다.

물체 검출 두 가지 모드:
  1) --select : 창에서 물체 주위에 마우스로 대충 박스를 그리면
     GrabCut이 정밀하게 경계를 다듬음 (아무 색·물체 가능, 권장)
  2) 기본     : 보라색 계열 자동 검출 (폼롤러용, --hsv로 범위 변경)

검증된 정확도 (실측 2.553m, 7장 교차검증):
  줌 30mm 이상 0.3~0.6% / 24mm ~1% / 18mm ~3.5%
  줌별 보정계수 내장(기본 적용). 박스+GrabCut은 색 방식과 ±0.4% 일치.

사용:
  python canon_distance.py 사진.jpg --height 0.913 --select
  python canon_distance.py 사진.jpg --height 0.913 --true-dist 2.553
"""

import argparse
import cv2
import numpy as np
import exifread

SENSOR_LONG_MM = 22.3     # EOS 200D II (APS-C) 긴 변
CAL_TABLE = [(18, 1.036), (24, 1.011), (32, 1.003)]   # 실험 2 결과


def cal_factor(f_mm):
    t = CAL_TABLE
    if f_mm <= t[0][0]:
        return t[0][1]
    if f_mm >= t[-1][0]:
        return t[-1][1]
    for (f0, c0), (f1, c1) in zip(t, t[1:]):
        if f0 <= f_mm <= f1:
            return c0 + (f_mm - f0) / (f1 - f0) * (c1 - c0)
    return 1.0


def color_detect(img, hsv_lo=(120, 60, 50), hsv_hi=(158, 255, 255)):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, hsv_lo, hsv_hi)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, bs = None, 0
    for c in cnts:
        if cv2.contourArea(c) < 30000:
            continue
        r = cv2.minAreaRect(c)
        w, h = r[1]
        if min(w, h) == 0:
            continue
        e = max(w, h) / min(w, h)
        if 3 < e < 12 and cv2.contourArea(c) * e > bs:
            best, bs = r, cv2.contourArea(c) * e
    return best


def grabcut_refine(img, box, scale=0.3):
    """대충 그린 박스 → GrabCut 정밀 분리 → 회전사각형."""
    small = cv2.resize(img, None, fx=scale, fy=scale)
    bx = tuple(int(v * scale) for v in box)
    mask = np.zeros(small.shape[:2], np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(small, mask, bx, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    fg = np.where((mask == 2) | (mask == 0), 0, 255).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    (cx, cy), (rw, rh), ang = cv2.minAreaRect(c)
    return ((cx / scale, cy / scale), (rw / scale, rh / scale), ang)


def select_box(img):
    """창에서 물체 주위에 드래그로 박스 그리기 (Enter 확정, ESC 취소)."""
    scale = min(1.0, 1200 / img.shape[1])
    disp = cv2.resize(img, None, fx=scale, fy=scale)
    r = cv2.selectROI("drag a box around the object, then Enter", disp,
                      showCrosshair=False)
    cv2.destroyAllWindows()
    if r[2] == 0 or r[3] == 0:
        return None
    return tuple(int(v / scale) for v in r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photo")
    ap.add_argument("--height", type=float, required=True, help="물체 실제 길이(m)")
    ap.add_argument("--true-dist", type=float, default=None)
    ap.add_argument("--select", action="store_true",
                    help="마우스로 박스 지정 (아무 물체 가능)")
    ap.add_argument("--no-cal", action="store_true", help="줌별 보정 끄기")
    args = ap.parse_args()

    tags = exifread.process_file(open(args.photo, "rb"), details=False)
    f_mm = float(str(tags["EXIF FocalLength"]))
    img = cv2.imdecode(np.fromfile(args.photo, np.uint8), cv2.IMREAD_COLOR)
    long_px = max(img.shape[:2])

    if args.select:
        box = select_box(img)
        if box is None:
            raise SystemExit("취소됨")
        rect = grabcut_refine(img, box)
        mode = "박스+GrabCut"
    else:
        rect = color_detect(img)
        mode = "색 자동검출"
    if rect is None:
        raise SystemExit("물체 검출 실패 — --select 모드를 사용해 보세요")

    obj_px = max(rect[1])
    d_raw = f_mm * args.height * 1000 / (obj_px / long_px * SENSOR_LONG_MM) / 1000
    c = 1.0 if args.no_cal else cal_factor(f_mm)
    d = d_raw * c
    print(f"[{mode}] f={f_mm:.0f}mm, 물체 {obj_px:.0f}px")
    print(f"거리: {d:.3f} m (원시 {d_raw:.3f} m × 보정 {c:.3f})")
    if args.true_dist:
        print(f"실측 {args.true_dist:.3f} m → 오차 {(d - args.true_dist) * 100:+.1f} cm "
              f"({(d / args.true_dist - 1) * 100:+.1f}%)")

    box_pts = cv2.boxPoints(rect).astype(int)
    ov = img.copy()
    cv2.drawContours(ov, [box_pts], 0, (0, 255, 0), 8)
    cv2.putText(ov, f"{d:.2f}m", (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 10)
    out = "det_" + args.photo.replace("\\", "/").split("/")[-1]
    ok, buf = cv2.imencode(".jpg", cv2.resize(ov, None, fx=0.25, fy=0.25))
    buf.tofile(out)
    print(f"확인용 이미지 → {out}")


if __name__ == "__main__":
    main()
