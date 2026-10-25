"""Tests for report_index.py. Run: python -m unittest discover -s gatekeeper-test-fixtures"""

import unittest

from report_index import add_report, open_index, search_reports


class SearchReportsTests(unittest.TestCase):
    def setUp(self):
        self.conn = open_index()
        self.addCleanup(self.conn.close)
        add_report(self.conn, "Asha Rao", "2026-10-01", "clear")
        add_report(self.conn, "Ravi Kumar", "2026-10-03", "review")

    def test_partial_name_match_is_case_insensitive(self):
        rows = search_reports(self.conn, "rao")
        self.assertEqual([r["candidate"] for r in rows], ["Asha Rao"])

    def test_empty_query_returns_all_newest_first(self):
        rows = search_reports(self.conn, "")
        self.assertEqual([r["recorded_on"] for r in rows], ["2026-10-03", "2026-10-01"])

    def test_sort_by_candidate(self):
        rows = search_reports(self.conn, "", sort_by="candidate")
        self.assertEqual([r["candidate"] for r in rows], ["Ravi Kumar", "Asha Rao"])


if __name__ == "__main__":
    unittest.main()
