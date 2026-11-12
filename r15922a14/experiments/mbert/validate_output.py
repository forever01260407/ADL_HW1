from __future__ import annotations

import argparse

from common import read_json, validate_prediction_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate ADL HW1 prediction.csv")
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--prediction-file", required=True)
    args = parser.parse_args()

    test_records = read_json(args.test_file)
    if not isinstance(test_records, list):
        raise ValueError("test.json must be a list")
    expected_ids = [record["id"] for record in test_records]
    validate_prediction_csv(args.prediction_file, expected_ids)
    print(f"OK: {len(expected_ids)} rows, exact header/order, binary values only.")


if __name__ == "__main__":
    main()

