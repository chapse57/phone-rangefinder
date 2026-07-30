# -*- coding: utf-8 -*-
"""
카메라 거리 측정 시뮬레이션 — 반복 측정으로 오차 수렴 검증
============================================================
ArUco 마커를 가상 카메라로 렌더링한 합성 이미지에서 거리를 추정하고,
반복 측정 + 이상치 제거 + 평균으로 오차가 줄어드는 과정을 실험한다.

구조:
  1. VirtualCamera  : 핀홀 카메라 모델 (내부 파라미터)
  2. render_scene() : 지정 거리/자세의 마커를 투영한 합성 이미지 생성
                      (센서 노이즈, 블러, 조명 변화, 자세 흔들림 포함)
  3. measure_once() : ArUco 검출 → solvePnP → 거리 1회 추정
  4. Estimator      : 반복 측정을 모아 IQR 이상치 제거 + 평균으로 수렴
  5. run_experiment(): 여러 거리에서 N회 반복, 오차 수렴 곡선 기록

실행: python3 distance_sim.py
출력: results/ 폴더에 그래프(png), 샘플 이미지, 결과 CSV
"""

import os
import csv
import numpy as np
import cv2

rng = np.random.default_rng(42)

# ---------------------------------------------------------------
# 1. 가상 카메라 (일반적인 웹캠 사양과 유사하게 설정)
# ---------------------------------------------------------------
IMG_W, IMG_H = 1280, 720
FX = FY = 950.0                      # 초점거리 (픽셀)
CX, CY = IMG_W / 2, IMG_H / 2
K = np.array([[FX, 0, CX],
              [0, FY, CY],
              [0,  0,  1]], dtype=np.float64)
DIST = np.zeros(5)                   # 측정 시 사용하는 왜곡 계수(무왜곡 가정)
# 실제 렌더링에는 약간의 미보정 왜곡을 넣어 "현실과 모델의 차이"를 재현
TRUE_DIST_COEFFS = np.array([0.03, -0.01, 0.0005, -0.0005, 0.0])

MARKER_SIZE = 0.10                   # 마커 한 변 10 cm
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
MARKER_ID = 7
MARKER_PX = 200
MARKER_IMG = cv2.aruco.generateImageMarker(ARUCO_DICT, MARKER_ID, MARKER_PX)

# 마커의 3D 코너 좌표 (마커 중심 기준, z=0 평면)
# solvePnP(IPPE_SQUARE)용: ArUco 관례 순서 (TL, TR, BR, BL / y축 위쪽)
half = MARKER_SIZE / 2
OBJ_PTS = np.array([[-half,  half, 0],
                    [ half,  half, 0],
                    [ half, -half, 0],
                    [-half, -half, 0]], dtype=np.float64)
# 렌더링용: 카메라 좌표계(y축 아래쪽)에서 이미지의 TL, TR, BR, BL에
# 대응하는 순서. 이걸 안 맞추면 마커가 거울상으로 그려져 검출 실패.
RENDER_PTS = np.array([[-half, -half, 0],
                       [ half, -half, 0],
                       [ half,  half, 0],
                       [-half,  half, 0]], dtype=np.float64)


# ---------------------------------------------------------------
# 2. 합성 이미지 렌더링
# ---------------------------------------------------------------
def render_scene(distance_m, jitter=True):
    """지정 거리에 마커를 두고 카메라로 찍은 것 같은 이미지를 만든다.

    현실성을 위해 매 프레임:
      - 카메라/손 흔들림: 위치 ±3 mm, 회전 ±3°
      - 미보정 렌즈 왜곡 (측정 모델은 모름)
      - 센서 노이즈(가우시안), 모션 블러, 조명 변화
    """
    # 마커 자세 (카메라 좌표계)
    tvec = np.array([0.0, 0.0, distance_m])
    rvec = np.array([0.0, 0.0, 0.0])
    if jitter:
        tvec += rng.normal(0, 0.003, 3)              # ±3 mm 흔들림
        rvec += np.deg2rad(rng.normal(0, 3.0, 3))    # ±3° 회전

    # 3D 코너 → 이미지 투영 (실제 왜곡 포함)
    img_pts, _ = cv2.projectPoints(RENDER_PTS, rvec, tvec, K, TRUE_DIST_COEFFS)
    img_pts = img_pts.reshape(-1, 2).astype(np.float32)

    # 배경 (조명 변화)
    bg = int(rng.uniform(140, 200))
    img = np.full((IMG_H, IMG_W), bg, dtype=np.uint8)

    # 마커 워프하여 합성
    src = np.array([[0, 0], [MARKER_PX, 0],
                    [MARKER_PX, MARKER_PX], [0, MARKER_PX]], dtype=np.float32)
    H_mat = cv2.getPerspectiveTransform(src, img_pts)
    warped = cv2.warpPerspective(MARKER_IMG, H_mat, (IMG_W, IMG_H),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_TRANSPARENT,
                                 dst=img.copy())
    mask = cv2.warpPerspective(np.full_like(MARKER_IMG, 255), H_mat,
                               (IMG_W, IMG_H))
    img[mask > 127] = warped[mask > 127]

    # 조명 스케일 + 블러 + 센서 노이즈
    gain = rng.uniform(0.85, 1.1)
    img = np.clip(img.astype(np.float32) * gain, 0, 255)
    ksize = int(rng.choice([1, 3, 3, 5]))
    if ksize > 1:
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)
    img += rng.normal(0, rng.uniform(2, 8), img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------
# 3. 1회 측정: 검출 → solvePnP → 거리
# ---------------------------------------------------------------
_params = cv2.aruco.DetectorParameters()
# 서브픽셀 코너 보정: 검출 편향(바이어스)을 크게 줄여
# 반복 평균이 실제로 효과를 내도록 함
_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT, _params)

def measure_once(img):
    """이미지에서 마커까지의 거리(m)를 추정. 실패 시 None."""
    corners, ids, _ = DETECTOR.detectMarkers(img)
    if ids is None or MARKER_ID not in ids:
        return None
    idx = list(ids.flatten()).index(MARKER_ID)
    pts = corners[idx].reshape(-1, 2).astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(OBJ_PTS, pts, K, DIST,
                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    return float(np.linalg.norm(tvec))


# ---------------------------------------------------------------
# 4. 반복 측정 추정기: IQR 이상치 제거 + 평균
# ---------------------------------------------------------------
class Estimator:
    def __init__(self):
        self.samples = []

    def add(self, d):
        if d is not None:
            self.samples.append(d)

    def estimate(self):
        """이상치를 걸러낸 뒤 평균. 표본 4개 미만이면 단순 평균/중앙값."""
        s = np.asarray(self.samples)
        if len(s) == 0:
            return None
        if len(s) < 4:
            return float(np.median(s))
        q1, q3 = np.percentile(s, [25, 75])
        iqr = q3 - q1
        keep = s[(s >= q1 - 1.5 * iqr) & (s <= q3 + 1.5 * iqr)]
        return float(keep.mean()) if len(keep) else float(np.median(s))


# ---------------------------------------------------------------
# 5. 실험: 거리별 N회 반복, 수렴 곡선 기록
# ---------------------------------------------------------------
def run_experiment(distances=(0.5, 1.0, 2.0, 3.0), n_iter=60, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    results = {}          # dist -> (수렴 곡선 오차[mm] 리스트, 원시 측정값)

    for d_true in distances:
        est = Estimator()
        raw, curve = [], []
        for i in range(n_iter):
            img = render_scene(d_true)
            m = measure_once(img)
            est.add(m)
            raw.append(m)
            cur = est.estimate()
            curve.append(abs(cur - d_true) * 1000 if cur else np.nan)
            if i == 0:  # 샘플 이미지 저장
                cv2.imwrite(f"{out_dir}/sample_{d_true:.1f}m.png", img)
        results[d_true] = (curve, raw)

        single_err = np.nanmean([abs(r - d_true) * 1000 for r in raw if r])
        print(f"[{d_true:.1f} m] 1회 측정 평균오차 {single_err:6.1f} mm  →  "
              f"{n_iter}회 반복 후 {curve[-1]:6.1f} mm")

    # CSV 저장
    with open(f"{out_dir}/measurements.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true_distance_m", "iteration", "raw_measure_m",
                    "converged_error_mm"])
        for d_true, (curve, raw) in results.items():
            for i, (r, c) in enumerate(zip(raw, curve)):
                w.writerow([d_true, i + 1, f"{r:.5f}" if r else "", f"{c:.2f}"])
    return results


def plot_results(results, out_dir="results"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (1) 반복 횟수 vs 오차 (수렴 곡선)
    for d_true, (curve, _) in results.items():
        axes[0].plot(range(1, len(curve) + 1), curve, label=f"{d_true:.1f} m")
    axes[0].set_xlabel("number of measurements")
    axes[0].set_ylabel("estimation error (mm)")
    axes[0].set_title("Error convergence with repeated measurements")
    axes[0].legend(title="true distance")
    axes[0].grid(alpha=0.3)

    # (2) 1회 측정 산포 vs 수렴값
    ds = list(results.keys())
    single = [np.nanstd([(r - d) * 1000 for r in results[d][1] if r])
              for d in ds]
    final = [results[d][0][-1] for d in ds]
    x = np.arange(len(ds))
    axes[1].bar(x - 0.18, single, 0.36, label="single-shot std (mm)")
    axes[1].bar(x + 0.18, final, 0.36, label="error after averaging (mm)")
    axes[1].set_xticks(x, [f"{d:.1f} m" for d in ds])
    axes[1].set_title("Single measurement vs. averaged estimate")
    axes[1].set_ylabel("mm")
    axes[1].legend()
    axes[1].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(f"{out_dir}/convergence.png", dpi=140)
    print(f"그래프 저장: {out_dir}/convergence.png")


if __name__ == "__main__":
    res = run_experiment()
    plot_results(res)
