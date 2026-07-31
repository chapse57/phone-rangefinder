# -*- coding: utf-8 -*-
"""
캐논 DSLR 사진에서 오토포커스 거리(FocusDistance) 읽기
========================================================
캐논 카메라는 촬영 시 렌즈가 초점을 맞춘 거리를 MakerNotes의
ShotInfo(태그 0x0004) 배열에 기록한다:
  index 19: FocusDistanceUpper (단위 0.01 m)
  index 20: FocusDistanceLower
(exiftool의 Canon.pm 정의와 동일)

사용: python canon_focus.py 사진.jpg [사진2.jpg ...]
필요: pip install exifread
"""

import sys
import exifread


def read_focus_distance(path):
    """(모델명, 렌즈, 초점거리mm, FD상한 m, FD하한 m) — 없으면 None 항목."""
    with open(path, "rb") as f:
        tags = exifread.process_file(f, details=True)

    model = str(tags.get("Image Model", "?"))
    lens = str(tags.get("EXIF LensModel", tags.get("MakerNote LensModel", "?")))
    focal = str(tags.get("EXIF FocalLength", "?"))

    upper = lower = None
    shot = tags.get("MakerNote Tag 0x0004")
    if shot is not None and hasattr(shot, "values") and len(shot.values) > 20:
        raw_u, raw_l = int(shot.values[19]), int(shot.values[20])
        # 0 또는 65535는 '기록 안 됨'
        if 0 < raw_u < 65535:
            upper = raw_u / 100.0
        if 0 < raw_l < 65535:
            lower = raw_l / 100.0
    return model, lens, focal, upper, lower


def main():
    if len(sys.argv) < 2:
        sys.exit("사용법: python canon_focus.py 사진.jpg")
    for p in sys.argv[1:]:
        print(f"\n=== {p} ===")
        try:
            model, lens, focal, up, lo = read_focus_distance(p)
        except Exception as e:
            print(f" 읽기 실패: {e}")
            continue
        print(f" 카메라: {model} / 렌즈: {lens} / 초점거리: {focal}mm")
        if up is None and lo is None:
            print(" FocusDistance 기록 없음 (렌즈가 거리 정보를 안 보내는 경우)")
        else:
            if up and lo:
                mid = (up + lo) / 2
                print(f" 초점 거리 기록: {lo:.2f} ~ {up:.2f} m  (중앙값 {mid:.2f} m)")
            else:
                print(f" 초점 거리 기록: {(up or lo):.2f} m")


if __name__ == "__main__":
    main()
