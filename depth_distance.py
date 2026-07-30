# -*- coding: utf-8 -*-
"""
AI 깊이 추정 거리 측정 — 기준물 없이 사진 한 장으로
======================================================
Depth Anything V2 (metric) 모델이 사진의 모든 픽셀에 대해 거리(m)를
추정한다. 마커도 카드도 필요 없다. 사진을 찍고, 알고 싶은 물체를
클릭하면 그 지점까지의 거리가 나온다.

정확도 개선 연구 루프:
  - AI의 초기 오차는 5~15% 수준 (장면·카메라에 따라 다름)
  - 실제 거리를 아는 사진으로 calibrate 하면 보정계수가 학습되어
    이후 측정에 자동 적용 → 실측을 쌓을수록 정확해짐
  - --true-dist 를 주면 오차가 experiment_log_ai.csv 에 누적 기록

명령:
  python depth_distance.py measure 사진.jpg            # 창에서 클릭 → 거리
  python depth_distance.py measure 사진.jpg --point 640 360   # 좌표 직접 지정
  python depth_distance.py measure 사진.jpg --true-dist 1.20  # 오차 기록
  python depth_distance.py calibrate 사진.jpg --true-dist 1.20 # 보정계수 학습

필요 패키지:
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  pip install transformers pillow opencv-contrib-python numpy

첫 실행 시 모델(~100MB)을 자동 다운로드한다. GPU 없어도 동작(수 초/장).
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime

import numpy as np
import cv2

SCALE_JSON = "depth_scale.json"
LOG_CSV = "experiment_log_ai.csv"
MODELS = {
    "indoor":  "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    "outdoor": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
}

_pipe = None


def get_pipe(scene):
    global _pipe
    if _pipe is None:
        print(f"AI 모델 로딩 중... ({MODELS[scene]})")
        from transformers import pipeline
        _pipe = pipeline("depth-estimation", model=MODELS[scene])
    return _pipe


def imread_unicode(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, img):
    ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".png", img)
    if ok:
        buf.tofile(path)


def predict_depth(path, scene):
    """사진 → 픽셀별 거리(m) 2D 배열."""
    from PIL import Image
    pil = Image.open(path).convert("RGB")
    out = get_pipe(scene)(pil)
    depth = np.array(out["predicted_depth"])
    if depth.ndim == 3:
        depth = depth[0]
    # 원본 해상도로 리사이즈
    img = imread_unicode(path)
    depth = cv2.resize(depth, (img.shape[1], img.shape[0]))
    return depth, img


def get_scale():
    """실측 보정계수 (없으면 1.0)."""
    if os.path.exists(SCALE_JSON):
        with open(SCALE_JSON, encoding="utf-8") as f:
            d = json.load(f)
        return d["scale"], d["n_samples"]
    return 1.0, 0


def point_depth(depth, x, y, win=7):
    """클릭 지점 주변 win x win 의 중앙값 (한 픽셀 노이즈 방지)."""
    h, w = depth.shape
    x0, x1 = max(0, x - win // 2), min(w, x + win // 2 + 1)
    y0, y1 = max(0, y - win // 2), min(h, y + win // 2 + 1)
    return float(np.median(depth[y0:y1, x0:x1]))


def pick_point(img, depth, scale):
    """창을 띄워 클릭 → 거리 표시. 여러 지점 클릭 가능, ESC/q로 종료."""
    disp_scale = min(1.0, 1200 / img.shape[1])
    base = cv2.resize(img, None, fx=disp_scale, fy=disp_scale)
    canvas = base.copy()
    picked = []

    def on_mouse(ev, x, y, flags, param):
        nonlocal canvas
        if ev == cv2.EVENT_LBUTTONDOWN:
            ox, oy = int(x / disp_scale), int(y / disp_scale)
            d = point_depth(depth, ox, oy) * scale
            picked.append((ox, oy, d))
            cv2.circle(canvas, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(canvas, f"{d:.2f} m", (x + 10, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            print(f" 클릭 ({ox},{oy}) → {d:.3f} m")

    win = "click object to get distance (q/ESC to finish)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        cv2.imshow(win, canvas)
        k = cv2.waitKey(20) & 0xFF
        if k in (27, ord('q')):
            break
    cv2.destroyWindow(win)
    return picked


def save_depth_vis(depth, img, path, scale, points=()):
    """깊이맵 컬러 시각화 저장 — AI가 장면을 어떻게 봤는지 확인용."""
    dmin, dmax = np.percentile(depth, [2, 98])
    norm = np.clip((depth - dmin) / max(dmax - dmin, 1e-6), 0, 1)
    vis = cv2.applyColorMap((255 * (1 - norm)).astype(np.uint8),
                            cv2.COLORMAP_TURBO)
    vis = cv2.addWeighted(img, 0.35, vis, 0.65, 0)
    for (x, y, d) in points:
        cv2.circle(vis, (x, y), 8, (255, 255, 255), 2)
        cv2.putText(vis, f"{d:.2f}m", (x + 10, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    out = os.path.splitext(path)[0] + "_depth.png"
    imwrite_unicode(out, vis)
    print(f" 깊이맵 저장 → {out}  (가까움=빨강, 멂=파랑)")


def cmd_measure(args):
    scale, n_cal = get_scale()
    depth, img = predict_depth(args.photo, args.scene)
    tag = f"보정계수 {scale:.3f} (실측 {n_cal}회 기반)" if n_cal else \
          "보정 안 됨 (calibrate 하면 정확도 향상)"
    print(f"분석 완료. {tag}")

    if args.point:
        x, y = args.point
        d = point_depth(depth, x, y) * scale
        picked = [(x, y, d)]
        print(f" 지점 ({x},{y}) → {d:.3f} m")
    else:
        picked = pick_point(img, depth, scale)
        if not picked:
            # GUI가 없거나 클릭 안 함 → 중앙점 자동 사용
            h, w = depth.shape
            d = point_depth(depth, w // 2, h // 2) * scale
            picked = [(w // 2, h // 2, d)]
            print(f" (클릭 없음 → 화면 중앙 사용) {d:.3f} m")
    save_depth_vis(depth, img, args.photo, scale, picked)

    if args.true_dist and picked:
        d = picked[-1][2]
        err = (d - args.true_dist) * 1000
        pct = abs(err) / args.true_dist / 10
        print(f" 실제 {args.true_dist:.3f} m → 오차 {err:+.0f} mm ({pct:.1f}%)")
        new = not os.path.exists(LOG_CSV)
        with open(LOG_CSV, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["시각", "사진", "실제_m", "추정_m", "오차_mm",
                            "오차_%", "보정계수", "보정실측수"])
            w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M"),
                        os.path.basename(args.photo), f"{args.true_dist:.3f}",
                        f"{d:.3f}", f"{err:+.0f}", f"{pct:.1f}",
                        f"{scale:.3f}", n_cal])
        print(f" 일지 기록 → {LOG_CSV}")


def cmd_calibrate(args):
    """실제 거리를 아는 지점으로 보정계수(실제/AI추정) 학습."""
    depth, img = predict_depth(args.photo, args.scene)
    if args.point:
        x, y = args.point
    else:
        print("보정할 지점(실제 거리를 아는 물체)을 클릭하세요.")
        picked = pick_point(img, depth, 1.0)
        if not picked:
            h, w = depth.shape
            x, y = w // 2, h // 2
        else:
            x, y = picked[-1][0], picked[-1][1]
    raw = point_depth(depth, x, y)
    ratio = args.true_dist / raw
    # 기존 보정과 합치기 (이동 평균)
    if os.path.exists(SCALE_JSON):
        with open(SCALE_JSON, encoding="utf-8") as f:
            prev = json.load(f)
        ratios = prev.get("ratios", []) + [ratio]
    else:
        ratios = [ratio]
    scale = float(np.median(ratios))
    with open(SCALE_JSON, "w", encoding="utf-8") as f:
        json.dump({"scale": scale, "ratios": ratios,
                   "n_samples": len(ratios),
                   "at": datetime.now().isoformat()}, f, indent=2)
    print(f"AI 원시 추정 {raw:.3f} m, 실제 {args.true_dist:.3f} m "
          f"→ 이번 비율 {ratio:.3f}")
    print(f"누적 보정계수 {scale:.3f} (실측 {len(ratios)}회) → {SCALE_JSON}")
    print("이후 measure가 자동 적용합니다. 실측을 쌓을수록 정확해집니다.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure", help="사진에서 클릭 지점 거리 측정")
    m.add_argument("photo")
    m.add_argument("--point", nargs=2, type=int, default=None,
                   metavar=("X", "Y"))
    m.add_argument("--true-dist", type=float, default=None)
    m.add_argument("--scene", choices=["indoor", "outdoor"], default="indoor")
    m.set_defaults(func=cmd_measure)
    c = sub.add_parser("calibrate", help="실측 거리로 보정계수 학습")
    c.add_argument("photo")
    c.add_argument("--true-dist", type=float, required=True)
    c.add_argument("--point", nargs=2, type=int, default=None,
                   metavar=("X", "Y"))
    c.add_argument("--scene", choices=["indoor", "outdoor"], default="indoor")
    c.set_defaults(func=cmd_calibrate)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
