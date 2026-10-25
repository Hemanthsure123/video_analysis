"""Tests for report_endpoint.py. Run: python -m unittest discover -s gatekeeper-test-fixtures"""

import json
import os
import tempfile
import unittest
from unittest import mock

import report_endpoint
from report_endpoint import get_report


class GetReportTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.report_dir = tmp.name
        patcher = mock.patch.object(report_endpoint, "REPORT_DIR", self.report_dir)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_returns_report(self):
        with open(os.path.join(self.report_dir, "0123456789ab.json"), "w", encoding="utf-8") as fh:
            json.dump({"verdict": "clear"}, fh)
        status, body = get_report({"params": {"job_id": "0123456789ab"}})
        self.assertEqual(status, 200)
        self.assertEqual(body, {"job_id": "0123456789ab", "report": {"verdict": "clear"}})

    def test_rejects_malformed_job_id(self):
        status, _ = get_report({"params": {"job_id": "../../.env"}})
        self.assertEqual(status, 400)

    def test_failure_returns_500_with_error(self):
        status, body = get_report({"params": {"job_id": "ffffffffffff"}})
        self.assertEqual(status, 500)
        self.assertIn("error", body)


if __name__ == "__main__":
    unittest.main()
