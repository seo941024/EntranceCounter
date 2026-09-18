# 실습 5 — Object Tracking : 출입 인원 카운터

슬라이드 추천 프로젝트 1번. 가상 선을 하나 긋고, 그 선을 넘는 사람을 방향별로 센다.
소스: `entrance_counter.py`. Ultralytics YOLO + BoT-SORT(`model.track(..., persist=True)`)로
프레임 간 사람 ID를 유지하고, ID별 중심점 궤적을 그린다.

## 설치

```bash
conda activate vision_new
pip install ultralytics opencv-contrib-python pillow
```

폰트는 `fonts/Pretendard-Regular.ttf`, `fonts/Pretendard-Bold.ttf`로 이미 같이 들어있음 (SIL OFL 라이선스, 별도 설치 불필요).
화면 텍스트는 OpenCV 기본 폰트가 아니라 Pretendard로 렌더링됨 (PIL을 거쳐서 그림).

## 데모

| 실행 화면 | 종료 후 콘솔 결과 |
|---|---|
| ![tracking demo](docs/demo_tracking.png) | ![console result](docs/demo_console.png) |

노란 세로선이 가상 라인, 사람마다 초록 박스 + `ID <번호>` + 주황색 이동 궤적이 표시된다.
종료(`q` 또는 영상 끝)하면 콘솔에 방향별 통과 인원과 전체 감지 ID 수가 정리되어 출력된다.

## 실행

```bash
# sample.mp4 (실습2_OVD에서 가져온 보행자 영상) 로 실행 - 기본값
python entrance_counter.py

# 웹캠으로 실행
python entrance_counter.py --source 0

# 가상 선 위치/방향 조정 (0.0~1.0 비율)
python entrance_counter.py --line-pos 0.4 --orientation vertical
```

화면에:
- 사람마다 박스 + `ID <번호>`
- 최근 30프레임 이동 궤적(주황색 선)
- 가상 선(노란색), 이 선을 넘으면 콘솔에 로그 + 화면 상단 카운트(`L->R`, `R->L`) 갱신

`q`로 종료하면 콘솔에 최종 통과 인원과 전체 감지 ID 수가 출력됨.

## 필수 요건 체크

| 요건 | 대응 |
|---|---|
| 프레임 간 동일 객체 ID 유지 추적기 | BoT-SORT (`model.track(..., persist=True, tracker="tracker_cfg/botsort_custom.yaml")`) |
| 궤적·속도·체류시간 중 1개 이상 계산 | 이동 궤적(ID별 중심점 30프레임) 시각화 |
| 가려짐·재등장 테스트 및 기록 | 아래 "가려짐 테스트 기록" 참고 |
| 영상·웹캠에서 ID·경로가 보이는 데모 | `entrance_counter.py` |

## 가려짐 테스트 기록 (포트폴리오용 — 직접 실행해서 관찰한 내용 채워넣기)

`sample.mp4`에서 두 사람이 겹치거나(예: 우산 쓴 사람들이 스쳐 지나갈 때) 한 사람이 화면 밖으로 나갔다가
다시 들어오는 장면을 찾아서 아래를 채운다:

| 상황 | 관찰 결과 |
|---|---|
| 두 사람이 겹쳐 지나감 | (ID가 유지됐는지 / 바뀌었는지 적기) |
| 화면 밖으로 나갔다가 재등장 | (같은 ID로 돌아왔는지 / 새 ID가 붙었는지 적기) |
| 카운트에 영향 있었는지 | (오탐/누락 있었는지 적기) |

BoT-SORT는 SORT/DeepSORT보다 카메라 움직임 보정이 들어가 있어 ID 스위치가 적은 편이지만,
완전히 막지는 못한다. 이 표를 "잘 되는 장면 vs 깨지는 장면" 비교 자료로 그대로 포트폴리오에 쓸 수 있다.

## 트래커 설정 (`tracker_cfg/botsort_custom.yaml`)

기본 BoT-SORT(`track_buffer=30`)를 그대로 쓰면 가려짐 후 재등장 시 ID가 자주 바뀐다.
`track_buffer`(가려진 트랙을 몇 프레임까지 살려둘지)를 늘리면 가려짐엔 강해지지만,
사람이 많은 영상일수록 살아있는 트랙 수가 늘어나 **속도가 느려지는 트레이드오프**가 있다
(직접 90까지 올려서 테스트해봤더니 눈에 띄게 느려져서, 45 정도로 절충함).

```yaml
track_buffer: 45       # 기본 30보다 살짝 여유, 90처럼 크게 올리면 느려짐
new_track_thresh: 0.3  # 기본 0.25보다 올려서 오검출로 인한 새 ID 생성을 줄임
```

## 확장 아이디어

- `--classes 0 2`처럼 사람(0)뿐 아니라 차량(2)도 같이 카운트
- 가상 선을 2개 이상 둬서 구역별 진입/이탈 계산
- 통과 로그를 CSV로 쌓아 시간대별 리포트 생성
- ByteTrack(`tracker="bytetrack.yaml"`)으로 바꿔서 ID 스위치 횟수 비교
