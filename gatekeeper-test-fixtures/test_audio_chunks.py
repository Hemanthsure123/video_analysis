"""Tests for audio_chunks.py. Run: python -m unittest discover -s gatekeeper-test-fixtures"""

import unittest

from audio_chunks import chunk_windows


class ChunkWindowsTests(unittest.TestCase):
    def test_windows_are_consecutive(self):
        self.assertEqual(chunk_windows(180, 60), [(0, 60), (60, 120), (120, 180)])

    def test_default_chunk_length(self):
        windows = chunk_windows(600)
        self.assertEqual(len(windows), 10)
        self.assertEqual(windows[-1], (540.0, 600.0))

    def test_empty_recording(self):
        self.assertEqual(chunk_windows(0), [])

    def test_rejects_non_positive_chunk_length(self):
        for chunk_sec in (0, -30):
            with self.subTest(chunk_sec=chunk_sec):
                with self.assertRaises(ValueError):
                    chunk_windows(120, chunk_sec)


if __name__ == "__main__":
    unittest.main()
