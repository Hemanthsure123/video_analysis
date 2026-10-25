"""Tests for timestamps.py. Run: python -m unittest discover -s gatekeeper-test-fixtures"""

import unittest

from timestamps import parse_ts


class ParseTsTests(unittest.TestCase):
    def test_minutes_and_seconds(self):
        self.assertEqual(parse_ts("00:00"), 0)
        self.assertEqual(parse_ts("02:05"), 125)
        self.assertEqual(parse_ts("7:30"), 450)
        self.assertEqual(parse_ts("59:59"), 3599)

    def test_hours(self):
        self.assertEqual(parse_ts("1:00:00"), 3600)
        self.assertEqual(parse_ts("2:03:04"), 7384)
        self.assertEqual(parse_ts("12:59:59"), 46799)

    def test_surrounding_whitespace_is_ignored(self):
        self.assertEqual(parse_ts("  01:02\n"), 62)

    def test_rejects_out_of_range_fields(self):
        for text in ("60:00", "00:60", "1:60:00", "1:00:60"):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    parse_ts(text)

    def test_rejects_malformed_input(self):
        cases = (
            "", "   ", "5", "1:2", "1:2:03", "-1:00", "+1:00", "01:02:03:04",
            "aa:bb", "1.5:00", "01 :02", "١٢:٣٠",
        )
        for text in cases:
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    parse_ts(text)

    def test_rejects_non_strings(self):
        for value in (None, 62, 62.0, b"01:02"):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    parse_ts(value)


if __name__ == "__main__":
    unittest.main()
