"""Tests for talk_time.py. Run: python -m unittest discover -s gatekeeper-test-fixtures"""

import unittest

from talk_time import talk_time_summary


class TalkTimeSummaryTests(unittest.TestCase):
    def test_two_speakers(self):
        turns = [
            {"start": 0.0, "end": 30.0, "speaker_label": "SPEAKER_00"},
            {"start": 30.0, "end": 40.0, "speaker_label": "SPEAKER_01"},
        ]
        summary = talk_time_summary(turns)
        self.assertEqual(summary["dominant_speaker"], "SPEAKER_00")
        self.assertEqual(summary["speakers"]["SPEAKER_00"]["share_pct"], 75.0)


if __name__ == "__main__":
    unittest.main()
