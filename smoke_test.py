"""Local smoke test for DiamondVision Worker 3.0."""

from __future__ import annotations

from pathlib import Path
import tempfile

import cv2
import numpy as np

import ocr
from processor import process_mobile_job


def main() -> None:
    ocr.read_jersey_number = lambda image, roster=None: ("", 0.0, "SMOKE_TEST")

    with tempfile.TemporaryDirectory(prefix="dv-smoke-") as temporary:
        root = Path(temporary)
        input_directory = root / "input"
        output_directory = root / "output"
        input_directory.mkdir()

        image = np.zeros((900, 1400, 3), dtype=np.uint8)
        cv2.rectangle(image, (250, 120), (1150, 820), (210, 210, 210), -1)
        cv2.putText(
            image,
            "12",
            (580, 520),
            cv2.FONT_HERSHEY_SIMPLEX,
            5,
            (20, 20, 20),
            18,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(input_directory / "sample.jpg"), image)

        summary = process_mobile_job(
            input_directory,
            output_directory,
            {
                "mode": "single",
                "team1": "Smoke Team",
                "roster1": "12 Test Player",
            },
        )

        report = output_directory / "Reports" / "diamondvision_report.csv"
        metadata = output_directory / "Reports" / "metadata.json"

        assert summary["total"] == 1, summary
        assert report.is_file(), report
        assert metadata.is_file(), metadata

        print("DiamondVision Worker 3.0 smoke test passed.")
        print(summary)


if __name__ == "__main__":
    main()
