# -*- coding: utf-8 -*-
"""
실습 5 - Object Tracking : 출입 인원 카운터
============================================

슬라이드 필수 요건
  1) 프레임 간 동일 객체에 ID를 유지하는 추적기를 쓸 것    -> Ultralytics YOLO + BoT-SORT (model.track)
  2) 궤적·속도·체류시간 중 1개 이상을 계산할 것             -> 화면에 각 ID의 이동 궤적(중심점 이어그리기) 표시
  3) 가려짐과 재등장 상황을 직접 테스트하고 기록할 것        -> 실행 중 관찰, README에 기록 (코드는 ID 유지 자체가 핵심)
  4) 영상·웹캠에서 ID와 경로가 보이는 데모일 것              -> 이 스크립트

핵심 아이디어
  화면에 가상의 선(수직/수평)을 하나 긋고, 각 사람(ID)의 중심점이
  프레임마다 그 선의 어느 쪽에 있는지 기록한다. "이전 프레임엔 왼쪽, 이번 프레임엔 오른쪽"처럼
  선을 넘는 순간을 잡아서 방향별로 카운트한다 (IN / OUT).

사용법
  python entrance_counter.py                         # sample.mp4 로 실행 (기본값)
  python entrance_counter.py --source 0              # 웹캠으로 실행
  python entrance_counter.py --source sample.mp4 --line-pos 0.5 --orientation vertical
"""

import argparse
import os
from collections import deque

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_REGULAR = os.path.join(HERE, "fonts", "Pretendard-Regular.ttf")
FONT_BOLD = os.path.join(HERE, "fonts", "Pretendard-Bold.ttf")

# cv2.putText는 OpenCV 내장 Hershey 폰트만 지원해서 커스텀 폰트(Pretendard)를 못 씀.
# PIL로 텍스트를 그린 뒤 다시 OpenCV(BGR) 이미지로 변환하는 방식으로 우회함.
_font_cache = {}


def _get_font(bold, size):
    key = (bold, size)
    if key not in _font_cache:
        path = FONT_BOLD if bold else FONT_REGULAR
        _font_cache[key] = ImageFont.truetype(path, size)
    return _font_cache[key]


def put_texts_pretendard(frame_bgr, items):
    """frame_bgr 위에 여러 텍스트를 Pretendard 폰트로 한 번에 그려서 반환.
    items: [{"text":..., "pos":(x,y), "size":24, "color_bgr":(0,0,255), "bold":False}, ...]
    사람 수만큼 매번 BGR<->RGB 변환을 반복하면 엄청 느려지므로(프레임당 변환 1회로 제한),
    반드시 프레임당 한 번만 호출해서 모든 텍스트를 몰아서 그린다.
    """
    img_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    for item in items:
        x, y = item["pos"]
        color_bgr = item.get("color_bgr", (255, 255, 255))
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
        font = _get_font(item.get("bold", False), item.get("size", 24))
        text = item["text"]
        stroke_w = item.get("stroke_width", 2)
        # PIL 내장 외곽선(stroke) 사용 - 어떤 배경 위에서도 잘 보이게 검은 테두리를 두껍게 줌
        draw.text((x, y), text, font=font, fill=color_rgb,
                  stroke_width=stroke_w, stroke_fill=(0, 0, 0))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def get_args():
    ap = argparse.ArgumentParser(description="Object Tracking 기반 출입 인원 카운터")
    ap.add_argument("--source", default=os.path.join(HERE, "sample.mp4"),
                     help="영상 파일 경로 또는 웹캠 번호(0)")
    ap.add_argument("--model", default="yolov8n.pt", help="YOLO 탐지 모델 (최초 실행 시 자동 다운로드)")
    ap.add_argument("--line-pos", type=float, default=0.5,
                     help="가상 선 위치 (0.0~1.0, 화면 폭/높이 기준 비율)")
    ap.add_argument("--orientation", choices=["vertical", "horizontal"], default="vertical",
                     help="가상 선 방향 - vertical: 세로선(좌우 통과 판정), horizontal: 가로선(상하 통과 판정)")
    ap.add_argument("--classes", type=int, nargs="+", default=[0],
                     help="추적할 클래스 id (COCO 기준 0=person). 여러 개 지정 가능")
    ap.add_argument("--max-width", type=int, default=960,
                     help="화면에 표시할 창의 최대 가로 폭(px). 원본이 더 크면 비율 유지하며 축소")
    ap.add_argument("--tracker", default=os.path.join(HERE, "tracker_cfg", "botsort_custom.yaml"),
                     help="트래커 설정 파일 (기본: track_buffer를 늘려 가려짐에 더 버티는 커스텀 설정)")
    return ap.parse_args()


def main():
    args = get_args()

    source = args.source
    if str(source).isdigit():
        source = int(source)

    model = YOLO(args.model)

    # 첫 프레임을 읽어서 화면 크기를 알아내고, 가상 선의 실제 픽셀 좌표를 계산
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"영상을 열 수 없습니다: {source}")
        return
    ok, frame = cap.read()
    if not ok:
        print("첫 프레임을 읽을 수 없습니다.")
        return
    h, w = frame.shape[:2]
    cap.release()

    if args.orientation == "vertical":
        line_coord = int(w * args.line_pos)   # 세로선의 x좌표
    else:
        line_coord = int(h * args.line_pos)   # 가로선의 y좌표

    # ID별 이전 위치(선 기준 어느 쪽인지) + 이동 궤적(최근 30개 점)
    prev_side = {}                  # {track_id: "L" or "R"}
    trails = {}                     # {track_id: deque of (x, y)}
    count_l_to_r = 0
    count_r_to_l = 0

    def side_of(cx, cy):
        pos = cx if args.orientation == "vertical" else cy
        return "L" if pos < line_coord else "R"

    print(f"[설정] source={args.source}  orientation={args.orientation}  line={args.line_pos}")
    print("가상 선을 넘는 방향에 따라 L->R / R->L 로 카운트합니다. 'q'로 종료.")

    # persist=True : 프레임이 바뀌어도 이전 프레임의 트랙(ID)을 이어서 씀
    # tracker: 기본 botsort.yaml 대신 track_buffer를 늘린 커스텀 설정을 사용 (가려짐 대응)
    results_gen = model.track(source=source, classes=args.classes, tracker=args.tracker,
                               persist=True, stream=True, verbose=False)

    # 화면 표시 축소 비율은 영상 내내 고정이므로 루프 밖에서 한 번만 계산
    display_scale = 1.0
    if args.max_width and w > args.max_width:
        display_scale = args.max_width / w

    for r in results_gen:
        frame = r.orig_img.copy()
        text_items = []  # 이번 프레임에 그릴 텍스트를 모아뒀다가 PIL 변환 딱 1번으로 처리

        # 가상 선 그리기
        if args.orientation == "vertical":
            cv2.line(frame, (line_coord, 0), (line_coord, h), (0, 255, 255), 2)
        else:
            cv2.line(frame, (0, line_coord), (w, line_coord), (0, 255, 255), 2)

        if r.boxes is not None and r.boxes.id is not None:
            ids = r.boxes.id.int().tolist()
            xyxy = r.boxes.xyxy.tolist()

            for track_id, box in zip(ids, xyxy):
                x1, y1, x2, y2 = box
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                # 궤적 갱신 (요건 2: 이동 궤적)
                trail = trails.setdefault(track_id, deque(maxlen=30))
                trail.append((cx, cy))

                # 선 통과 판정 (요건의 핵심: 방향별 카운트)
                cur_side = side_of(cx, cy)
                prev = prev_side.get(track_id)
                if prev is not None and prev != cur_side:
                    if prev == "L" and cur_side == "R":
                        count_l_to_r += 1
                        print(f"[통과] ID {track_id}: L -> R  (누적 L->R={count_l_to_r})")
                    elif prev == "R" and cur_side == "L":
                        count_r_to_l += 1
                        print(f"[통과] ID {track_id}: R -> L  (누적 R->L={count_r_to_l})")
                prev_side[track_id] = cur_side

                # 박스는 cv2로 바로 그림 (빠름). 텍스트는 나중에 한 번에 그리기 위해 목록에만 추가
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 2)
                text_items.append({
                    "text": f"ID {track_id}",
                    "pos": (int(x1 * display_scale), int((y1 - 26) * display_scale)),
                    "size": 20, "color_bgr": (0, 200, 0), "bold": True,
                })

                # 궤적 그리기
                pts = list(trail)
                for i in range(1, len(pts)):
                    cv2.line(frame, pts[i - 1], pts[i], (255, 140, 0), 2)

        # 카운트 표시도 목록에 추가 (흰 글씨 + 검정 굵은 테두리 - 배경 색과 무관하게 잘 보임)
        text_items.append({"text": f"L→R: {count_l_to_r}   R→L: {count_r_to_l}",
                            "pos": (12, 10), "size": 23, "color_bgr": (255, 255, 255),
                            "bold": True, "stroke_width": 2})
        text_items.append({"text": f"누적 추적 ID 수: {len(trails)}",
                            "pos": (12, 38), "size": 16, "color_bgr": (255, 255, 255),
                            "bold": True, "stroke_width": 1})

        # 화면 표시용으로만 축소 (좌표 계산은 전부 원본 해상도 기준이라 정확도엔 영향 없음)
        if display_scale != 1.0:
            frame = cv2.resize(frame, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_AREA)

        # 이 프레임의 텍스트를 전부 모아서 PIL 변환 1번으로 그림 (사람 수와 무관하게 항상 1회)
        frame = put_texts_pretendard(frame, text_items)

        cv2.imshow("Entrance Counter (Object Tracking)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print("\n[최종 결과]")
    print(f"  L -> R 통과: {count_l_to_r}명")
    print(f"  R -> L 통과: {count_r_to_l}명")
    print(f"  전체 감지된 ID 수: {len(trails)}")


if __name__ == "__main__":
    main()
