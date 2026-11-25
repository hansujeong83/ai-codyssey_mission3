"""Mini NPU simulator: loop-based MAC, Cross/X classification and reports."""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
from pathlib import Path
from typing import Any

EPSILON = 1e-9
SUPPORTED_SIZES = (3, 5, 13, 25)
DATA_PATH = Path(__file__).with_name("data.json")
BENCHMARK_REPEATS = 100
BENCHMARK_SAMPLES = 5
INTERACTIVE_REPEATS = 10


def normalize_label(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError(f"라벨은 문자열이어야 합니다: {value!r}")
    lowered = value.strip().lower()
    if lowered in {"+", "cross"}:
        return "Cross"
    if lowered == "x":
        return "X"
    if lowered in {"undecided", "tie", "draw"}:
        return "UNDECIDED"
    raise ValueError(f"지원하지 않는 라벨: {value!r}")


def validate_matrix(matrix: Any, expected_size: int | None = None, name: str = "matrix") -> int:
    if not isinstance(matrix, list) or not matrix:
        raise ValueError(f"{name}: 비어 있지 않은 2차원 배열이 필요합니다.")
    n = len(matrix)
    if expected_size is not None and n != expected_size:
        raise ValueError(f"{name}: 행 크기 {n}, 예상 크기 {expected_size}")
    for row_index, row in enumerate(matrix):
        if not isinstance(row, list) or len(row) != n:
            raise ValueError(f"{name}: {row_index + 1}행의 열 개수가 {n}이 아닙니다.")
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name}: 숫자가 아닌 값이 있습니다.")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name}: NaN 또는 무한대는 사용할 수 없습니다.")
    return n


def mac(pattern: list[list[float]], filt: list[list[float]]) -> float:
    """Return sum(pattern[i][j] * filt[i][j]) using two explicit loops."""
    n = validate_matrix(pattern, name="pattern")
    validate_matrix(filt, expected_size=n, name="filter")
    accumulator = 0.0
    for row in range(n):
        for column in range(n):
            accumulator += float(pattern[row][column]) * float(filt[row][column])
    return accumulator


def classify(pattern, cross_filter, x_filter, epsilon: float = EPSILON):
    if (not isinstance(epsilon, (int, float)) or isinstance(epsilon, bool)
            or epsilon < 0 or not math.isfinite(epsilon)):
        raise ValueError("epsilon은 0 이상의 유한한 숫자여야 합니다.")
    cross_score = mac(pattern, cross_filter)
    x_score = mac(pattern, x_filter)
    if abs(cross_score - x_score) < epsilon:
        label = "UNDECIDED"
    elif cross_score > x_score:
        label = "Cross"
    else:
        label = "X"
    return cross_score, x_score, label


def make_shape(size: int, label: str) -> list[list[int]]:
    if not isinstance(size, int) or isinstance(size, bool) or size < 1:
        raise ValueError("size는 1 이상의 정수여야 합니다.")
    if label == "Cross":
        return [[1 if row == size // 2 or col == size // 2 else 0
                 for col in range(size)] for row in range(size)]
    if label == "X":
        return [[1 if row == col or row + col == size - 1 else 0
                 for col in range(size)] for row in range(size)]
    raise ValueError("label은 Cross 또는 X여야 합니다.")


def make_demo_dataset() -> dict[str, Any]:
    """Create a local test fixture; it is not the course's official dataset."""
    filters, patterns = {}, {}
    for size in (5, 13, 25):
        filters[f"size_{size}"] = {
            "cross": make_shape(size, "Cross"),
            "x": make_shape(size, "X"),
        }
        cross = make_shape(size, "Cross")
        xshape = make_shape(size, "X")
        noisy_cross = [row[:] for row in cross]
        noisy_x = [row[:] for row in xshape]
        for row in range(size):
            for col in range(size):
                if cross[row][col] == 0 and xshape[row][col] == 1:
                    noisy_cross[row][col] = 0.05
                if xshape[row][col] == 0 and cross[row][col] == 1:
                    noisy_x[row][col] = 0.05
        patterns[f"size_{size}_1"] = {"input": cross, "expected": "+"}
        patterns[f"size_{size}_2"] = {"input": xshape, "expected": "x"}
        patterns[f"size_{size}_3"] = {"input": noisy_cross, "expected": "Cross"}
        patterns[f"size_{size}_4"] = {"input": noisy_x, "expected": "X"}
    return {
        "_notice": "SELF-CREATED TEST DATA based on the assignment specification; not represented as an instructor-provided original dataset.",
        "filters": filters,
        "patterns": patterns,
    }


def load_dataset(path: str | Path = DATA_PATH) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON 문법 오류: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("JSON 최상위 값은 객체여야 합니다.")
    if not isinstance(data.get("filters"), dict):
        raise ValueError("최상위 filters 객체가 필요합니다.")
    if not isinstance(data.get("patterns"), dict):
        raise ValueError("최상위 patterns 객체가 필요합니다.")
    if not data["patterns"]:
        raise ValueError("patterns에 최소 1개 케이스가 필요합니다.")
    return data


def extract_size(case_key: str) -> int:
    match = re.match(r"^size_(\d+)_", str(case_key))
    if not match:
        raise ValueError(f"케이스 키 형식 오류: {case_key!r} (예: size_5_1)")
    size = int(match.group(1))
    if size not in (5, 13, 25):
        raise ValueError(f"JSON에서 지원하지 않는 행렬 크기: {size}; 지원 크기=(5, 13, 25)")
    return size


def get_filter_pair(filters: dict[str, Any], size: int):
    entry = filters.get(f"size_{size}")
    if not isinstance(entry, dict):
        raise ValueError(f"filters.size_{size}가 없거나 객체가 아닙니다.")
    normalized = {}
    for key, matrix in entry.items():
        label = normalize_label(key)
        if label in normalized:
            raise ValueError(f"size_{size}: {label} 필터가 중복되었습니다.")
        validate_matrix(matrix, expected_size=size, name=f"filter {key}")
        normalized[label] = matrix
    if "Cross" not in normalized or "X" not in normalized:
        raise ValueError(f"size_{size}: Cross(또는 +)와 X 필터가 모두 필요합니다.")
    return normalized["Cross"], normalized["X"]


def evaluate_case(case_key: str, case: Any, filters: dict[str, Any]) -> dict[str, Any]:
    size = extract_size(case_key)
    if not isinstance(case, dict) or "input" not in case or "expected" not in case:
        raise ValueError("케이스에는 input과 expected가 필요합니다.")
    pattern = case["input"]
    validate_matrix(pattern, expected_size=size, name=f"{case_key}.input")
    expected = normalize_label(case["expected"])
    if expected == "UNDECIDED":
        raise ValueError("expected에는 Cross(+) 또는 X만 사용할 수 있습니다.")
    cross_filter, x_filter = get_filter_pair(filters, size)
    cross_score, x_score, predicted = classify(pattern, cross_filter, x_filter)
    passed = predicted == expected
    reason = "예상 라벨과 일치" if passed else f"예상={expected}, 실제={predicted}"
    return {
        "case": case_key, "size": size, "cross_score": cross_score,
        "x_score": x_score, "predicted": predicted, "expected": expected,
        "status": "PASS" if passed else "FAIL", "reason": reason,
    }


def analyze_dataset(path: str | Path = DATA_PATH, verbose: bool = True):
    data = load_dataset(path)
    results = []
    for case_key, case in data["patterns"].items():
        try:
            result = evaluate_case(case_key, case, data["filters"])
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            result = {"case": str(case_key), "status": "FAIL", "reason": str(exc)}
        results.append(result)
    passed = sum(item["status"] == "PASS" for item in results)
    failed = len(results) - passed
    report = {"total": len(results), "passed": passed, "failed": failed, "results": results}
    if verbose:
        print("\n=== [1] JSON 데이터 분석 ===")
        for item in results:
            if "predicted" in item:
                print(
                    f"{item['case']}: Cross={item['cross_score']:.8g}, "
                    f"X={item['x_score']:.8g}, 판정={item['predicted']}, "
                    f"expected={item['expected']} -> {item['status']} ({item['reason']})"
                )
            else:
                print(f"{item['case']}: FAIL ({item['reason']})")
        print(f"\n총 테스트: {len(results)} | 통과: {passed} | 실패: {failed}")
        if failed:
            print("실패 케이스:")
            for item in results:
                if item["status"] == "FAIL":
                    print(f"- {item['case']}: {item['reason']}")
    return report


def benchmark(repeats: int = BENCHMARK_REPEATS,
              samples: int = BENCHMARK_SAMPLES, verbose: bool = True):
    if repeats < 10 or samples < 1:
        raise ValueError("repeats는 최소 10, samples는 1 이상이어야 합니다.")
    rows = []
    for size in SUPPORTED_SIZES:
        pattern = make_shape(size, "Cross")
        filt = [[0.5 for _ in range(size)] for _ in range(size)]
        mac(pattern, filt)
        measurements = []
        for _ in range(samples):
            start = time.perf_counter()
            for _ in range(repeats):
                mac(pattern, filt)
            measurements.append((time.perf_counter() - start) * 1000 / repeats)
        rows.append({
            "size": size, "operations": size * size,
            "mean_ms": statistics.mean(measurements),
            "min_ms": min(measurements), "max_ms": max(measurements),
            "samples": samples, "repeats_per_sample": repeats,
        })
    if verbose:
        print("\n=== 성능 분석 (크기별 평균 MAC 시간) ===")
        print(f"{'크기(N×N)':>12} {'평균 시간(ms)':>16} {'연산 횟수(N²)':>16}")
        print("-" * 48)
        for row in rows:
            print(f"{row['size']}×{row['size']:<9} {row['mean_ms']:>16.6f} {row['operations']:>16}")
        print(f"측정 기준: 샘플 {samples}회, 샘플당 MAC {repeats}회 반복")
    return rows


def read_matrix(title: str, size: int):
    print(f"{title} ({size}줄 입력, 각 줄에 숫자 {size}개를 공백으로 구분)")
    matrix = []
    for row_index in range(size):
        while True:
            raw = input(f"{row_index + 1}행: ").strip()
            try:
                values = [float(part) for part in raw.split()]
                if len(values) != size or not all(math.isfinite(v) for v in values):
                    raise ValueError
                matrix.append(values)
                break
            except ValueError:
                print(f"입력 형식 오류: 각 줄에 유한한 숫자 {size}개를 공백으로 구분해 입력하세요.")
    return matrix


def interactive_mode():
    size = 3
    print("\n=== [사용자 입력 모드: 3×3] ===")
    filter_a = read_matrix("필터 A", size)
    filter_b = read_matrix("필터 B", size)
    print("\n[필터 저장 확인]")
    print("필터 A:", filter_a)
    print("필터 B:", filter_b)
    while True:
        confirm = input("입력한 필터를 사용하시겠습니까? (y/n): ").strip().lower()
        if confirm in {"y", "yes"}:
            break
        if confirm in {"n", "no"}:
            print("필터를 다시 입력합니다.")
            return interactive_mode()
        print("y 또는 n으로 입력해 주세요.")

    pattern = read_matrix("패턴", size)
    score_a = score_b = 0.0
    start = time.perf_counter()
    for _ in range(INTERACTIVE_REPEATS):
        score_a = mac(pattern, filter_a)
        score_b = mac(pattern, filter_b)
    elapsed_ms = (time.perf_counter() - start) * 1000 / INTERACTIVE_REPEATS
    if abs(score_a - score_b) < EPSILON:
        result = "판정 불가 (|A-B| < 1e-9)"
    else:
        result = "A" if score_a > score_b else "B"
    print("\n=== MAC 결과 ===")
    print(f"A 점수: {score_a:.12g}")
    print(f"B 점수: {score_b:.12g}")
    print(f"연산 시간(평균/{INTERACTIVE_REPEATS}회): {elapsed_ms:.6f} ms")
    print(f"판정: {result}")
    benchmark(repeats=BENCHMARK_REPEATS, samples=BENCHMARK_SAMPLES)


def main():
    parser = argparse.ArgumentParser(description="Mini NPU Cross/X classifier")
    parser.add_argument("--make-demo", action="store_true",
                        help="공식 원본이 아닌 데모 data.json 생성")
    parser.add_argument("--analyze", action="store_true", help="data.json 분석 및 성능 분석")
    parser.add_argument("--benchmark", action="store_true", help="크기별 성능 분석")
    parser.add_argument("--data", default=str(DATA_PATH), help="분석할 JSON 파일 경로")
    parser.add_argument("--repeats", type=int, default=BENCHMARK_REPEATS,
                        help="샘플당 MAC 반복 횟수(최소 10)")
    parser.add_argument("--samples", type=int, default=BENCHMARK_SAMPLES,
                        help="성능 측정 샘플 수")
    args = parser.parse_args()

    if args.make_demo:
        output = Path(args.data)
        output.write_text(json.dumps(make_demo_dataset(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"데모 데이터 생성: {output.resolve()}")
        return
    if args.analyze:
        try:
            report = analyze_dataset(args.data)
            benchmark(args.repeats, args.samples)
            if report["failed"]:
                raise SystemExit(1)
        except (OSError, ValueError) as exc:
            print(f"분석 오류: {exc}")
            raise SystemExit(2)
        return
    if args.benchmark:
        benchmark(args.repeats, args.samples)
        return

    print("=== Mini NPU Simulator ===")
    while True:
        print("\n[모드 선택]\n1. 사용자 입력 (3×3)\n2. data.json 분석\n3. 종료")
        try:
            choice = input("선택: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료합니다.")
            return
        if choice == "1":
            try:
                interactive_mode()
            except (EOFError, KeyboardInterrupt):
                print("\n입력이 취소되었습니다.")
        elif choice == "2":
            try:
                report = analyze_dataset(args.data)
                benchmark(args.repeats, args.samples)
                print(f"\n결과 요약: 총 {report['total']}개 / PASS {report['passed']}개 / FAIL {report['failed']}개")
            except (OSError, ValueError) as exc:
                print(f"데이터 분석 실패: {exc}")
        elif choice == "3":
            print("종료합니다.")
            return
        else:
            print("1~3 중에서 선택해 주세요.")


if __name__ == "__main__":
    main()
