# phone-rangefinder

스마트폰 사진 한 장으로 물체까지의 거리를 재는 도구. 인쇄한 ArUco 마커를 기준으로 삼아, 여러 장의 사진을 반복 측정하고 이상치를 제거·평균해 정확도를 끌어올린다. 실제 거리를 알려주면 오차가 실험 일지에 누적 기록되어, 연구하듯 정확도를 개선해 나갈 수 있다.

## 원리

크기를 아는 평면 마커(ArUco)를 카메라로 찍으면, 이미지 속 마커 네 꼭짓점의 위치로부터 `solvePnP`가 카메라–마커 사이의 3D 자세를 복원한다. 그 이동 벡터의 크기가 곧 거리다. 촬영 각도는 자세 추정이 자동 보정하므로 기울여 찍어도 된다.

정확도의 핵심은 초점거리(fx)를 정확히 아는 것인데, 이는 카메라 기종마다 다르다. 그래서 알려진 거리에서 한 번 캘리브레이션해 fx를 역산해 두면(`camera.json`), 이후 측정이 크게 정확해진다.

## 설치

```bash
pip install -r requirements.txt
```

## 사용법

**1. 마커 인쇄**
```bash
python phone_distance.py marker
```
`marker_print_A4.png`를 **실제 크기(100%)**로 인쇄하고, 시트의 눈금자가 자로 100mm인지 확인한 뒤 벽에 평평하게 붙인다.

**2. 촬영**
폰으로 마커를 5~10장 촬영해 `photos/` 폴더에 넣는다. (아이폰 HEIC는 JPG로 변환)

**3. 측정**
```bash
python phone_distance.py measure --photos photos
```

**4. 캘리브레이션 (정확도 대폭 향상, 폰마다 1회)**
줄자로 정확히 잰 거리에서 찍은 사진들로:
```bash
python phone_distance.py calibrate --photos photos --true-dist 1.00
```
`camera.json`이 생성되어 이후 측정에 자동 적용된다. 폰을 바꾸면 다시 실행.

**5. 오차 추적 (연구 루프)**
실제 거리를 함께 주면 매 실험의 오차가 `experiment_log.csv`에 쌓인다:
```bash
python phone_distance.py measure --photos photos --true-dist 1.00
```

## 변수 처리

| 변수 | 처리 방식 |
|------|-----------|
| 촬영 각도 | solvePnP 자세 추정이 자동 보정 |
| 카메라 기종 | `camera.json`(캘리브레이션) → 사진 EXIF → 기본 FOV 가정 순 자동 결정 |
| 마커 크기 | 기본 10cm. 다른 크기는 `--marker-size 0.15`로 지정(인쇄물과 일치 필수) |

## 검증

`sim/` 폴더의 `distance_sim.py`는 합성 이미지로 측정 파이프라인을 검증한다. 반복 측정이 거듭될수록 오차가 수렴하는 것을 보여준다(`convergence.png`).

## 라이선스

MIT
