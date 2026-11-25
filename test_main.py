import json
import math
import tempfile
import unittest
from pathlib import Path

from main import (
    EPSILON, analyze_dataset, benchmark, classify, evaluate_case, extract_size,
    get_filter_pair, load_dataset, mac, make_demo_dataset, make_shape,
    normalize_label, validate_matrix,
)


class LabelAndMatrixTests(unittest.TestCase):
    def test_normalize_labels(self):
        self.assertEqual(normalize_label("+"), "Cross")
        self.assertEqual(normalize_label("cross"), "Cross")
        self.assertEqual(normalize_label("X"), "X")
        self.assertEqual(normalize_label("undecided"), "UNDECIDED")

    def test_reject_unknown_label(self):
        with self.assertRaises(ValueError):
            normalize_label("circle")

    def test_validate_matrix_sizes(self):
        self.assertEqual(validate_matrix([[1, 0], [0, 1]]), 2)
        with self.assertRaises(ValueError):
            validate_matrix([[1, 0], [1]])
        with self.assertRaises(ValueError):
            validate_matrix([[1]], expected_size=3)

    def test_reject_boolean_and_nonfinite_values(self):
        for value in (True, float("nan"), float("inf"), "1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_matrix([[value]])

    def test_mac_known_result(self):
        self.assertEqual(mac([[1, 2], [3, 4]], [[5, 6], [7, 8]]), 70.0)

    def test_mac_rejects_mismatched_shapes(self):
        with self.assertRaises(ValueError):
            mac([[1, 2], [3, 4]], [[1]])

    def test_classification_cross_and_x(self):
        for size in (3, 5, 13, 25):
            cross = make_shape(size, "Cross")
            xshape = make_shape(size, "X")
            cross_filter, x_filter = cross, xshape
            self.assertEqual(classify(cross, cross_filter, x_filter)[2], "Cross")
            self.assertEqual(classify(xshape, cross_filter, x_filter)[2], "X")

    def test_epsilon_tie(self):
        self.assertEqual(classify([[1]], [[1]], [[1 + EPSILON / 2]])[2], "UNDECIDED")

    def test_epsilon_validation(self):
        with self.assertRaises(ValueError):
            classify([[1]], [[1]], [[0]], epsilon=-1)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "data.json"
        self.path.write_text(json.dumps(make_demo_dataset()), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_demo_data_all_pass(self):
        report = analyze_dataset(self.path, verbose=False)
        self.assertEqual(report["total"], 8)
        self.assertEqual(report["passed"], 8)
        self.assertEqual(report["failed"], 0)

    def test_extract_size(self):
        self.assertEqual(extract_size("size_13_4"), 13)
        with self.assertRaises(ValueError):
            extract_size("bad-key")
        with self.assertRaises(ValueError):
            extract_size("size_7_1")

    def test_filter_aliases(self):
        filters = {"size_3": {"+": make_shape(3, "Cross"), "x": make_shape(3, "X")}}
        cross, xshape = get_filter_pair(filters, 3)
        self.assertEqual(cross, make_shape(3, "Cross"))
        self.assertEqual(xshape, make_shape(3, "X"))

    def test_evaluate_case_reason(self):
        data = make_demo_dataset()
        case = data["patterns"]["size_5_1"]
        result = evaluate_case("size_5_1", case, data["filters"])
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["reason"], "예상 라벨과 일치")

    def test_malformed_case_is_reported_as_fail(self):
        data = make_demo_dataset()
        data["patterns"]["size_5_bad"] = {"input": [[1]], "expected": "X"}
        self.path.write_text(json.dumps(data), encoding="utf-8")
        report = analyze_dataset(self.path, verbose=False)
        self.assertEqual(report["failed"], 1)
        self.assertEqual(report["results"][-1]["status"], "FAIL")

    def test_invalid_json_reports_error(self):
        self.path.write_text("{", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_dataset(self.path)

    def test_invalid_top_level_reports_error(self):
        self.path.write_text("[]", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_dataset(self.path)


class BenchmarkTests(unittest.TestCase):
    def test_benchmark_reports_all_required_sizes(self):
        result = benchmark(repeats=2, samples=2, verbose=False)
        self.assertEqual([row["size"] for row in result], [3, 5, 13, 25])
        self.assertTrue(all(row["mean_ms"] >= 0 for row in result))
        self.assertTrue(all(row["operations"] == row["size"] ** 2 for row in result))

    def test_benchmark_rejects_zero_repeats(self):
        with self.assertRaises(ValueError):
            benchmark(repeats=0, verbose=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
