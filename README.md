# Mission 3 — Mini NPU Simulator

순수 Python 이중 반복문으로 MAC(Multiply-Accumulate)을 계산하고, Cross(십자가)와 X 패턴의 점수를 비교하는 미니 NPU 시뮬레이터입니다.

## 핵심 구현
1. `mac(pattern, filter)`: 같은 위치 원소를 곱하고 누적합니다.
2. `validate_matrix`: 정사각형, 동일 크기, 숫자형, 유한값을 검사합니다.
3. `normalize_label`: `+`/`cross`를 `Cross`, `x`를 `X`로 통일합니다.
4. `classify`: 두 점수 차이가 epsilon(`1e-9`) 미만이면 `UNDECIDED`로 판정합니다.
5. JSON 분석: `size_N_idx` 키에서 행렬 크기를 추출하고, 케이스별 PASS/FAIL 및 실패 이유를 남깁니다.
6. 성능 측정: 3×3, 5×5, 13×13, 25×25 각각 5개 샘플을 측정하고 평균·최소·최대 시간을 계산합니다.

## 실행
대화형:
```bash
python main.py
```

데이터 분석:
```bash
python main.py --analyze
```

성능 분석:
```bash
python main.py --benchmark
```

데이터 파일 경로 지정:
```bash
python main.py --analyze --data ./data.json
```

반복 횟수/샘플 수 지정:
```bash
python main.py --benchmark --repeats 2000 --samples 7
```

## 중요: 공식 원본 data.json
과제에서 요구한 원본 `data.json`이 현재 제공되지 않아, 저장소의 `data.json`은 프로그램 검증용 데모 fixture입니다. 공식 채점 데이터가 아닙니다. 원본 파일을 전달받으면 `data.json`을 교체한 뒤 `python main.py --analyze`를 다시 실행해야 합니다. 데모 결과를 원본 데이터의 검증 결과로 제출하지 않습니다.

데모 파일을 다시 생성하려면:
```bash
python main.py --make-demo
```

## JSON 구조
```json
{
  "filters": {
    "size_5": {
      "cross": [[0, 0, 1, 0, 0]],
      "x": [[1, 0, 0, 0, 1]]
    }
  },
  "patterns": {
    "size_5_1": {
      "input": [[0, 0, 1, 0, 0]],
      "expected": "+"
    }
  }
}
```
위 행렬은 문서에서 구조만 설명하기 위한 축약 예시입니다. 실제 입력은 크기에 맞는 N행 N열이어야 합니다. 지원 크기는 3, 5, 13, 25입니다.

## 알고리즘과 복잡도
각 MAC 호출은 N×N의 모든 원소를 한 번씩 방문합니다. 각 원소마다 곱셈 1회와 누적 덧셈 1회를 수행하므로 시간 복잡도는 O(N²), 추가 공간 복잡도는 O(1)입니다(입력 행렬 저장 공간 제외). N이 2배가 되면 원소 방문 수는 약 4배가 됩니다.

## 테스트
```bash
python -m py_compile main.py test_main.py
python -m unittest -v
```
테스트에는 알려진 MAC 결과, 4개 크기 분류, 라벨 별칭, epsilon 동점, 비정상 행렬, JSON 파싱 오류, 케이스 단위 실패, 벤치마크 크기 및 입력 검증이 포함됩니다.

## 결과 해석 주의
- `PASS`: 계산 결과 라벨과 JSON의 expected가 일치함
- `FAIL`: 라벨 불일치 또는 케이스 구조/행렬/필터/라벨 오류
- `UNDECIDED`: 두 점수 차이가 epsilon 미만인 동점 정책 결과
- 실행 시간은 CPU, Python 버전, OS, 백그라운드 부하에 따라 달라지므로 실행한 PC의 측정값을 기록해야 합니다.
