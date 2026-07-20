import unittest

import pandas as pd

from feedback_utils import (
    find_date_column,
    find_feedback_column,
    remove_blank_feedback,
)


class FeedbackUtilsTest(unittest.TestCase):
    def test_prefers_xquik_tweet_text_over_other_text_columns(self):
        frame = pd.DataFrame(
            {
                "author": ["Ada"],
                "Tweet Text": ["Useful release"],
            }
        )

        self.assertEqual(find_feedback_column(frame), "Tweet Text")

    def test_removes_null_empty_and_whitespace_feedback(self):
        frame = pd.DataFrame(
            {
                "tweet_text": [None, "", "  ", " useful "],
                "id": [1, 2, 3, 4],
            }
        )

        cleaned = remove_blank_feedback(frame, "tweet_text")

        self.assertEqual(cleaned["tweet_text"].tolist(), ["useful"])
        self.assertEqual(cleaned["id"].tolist(), [4])

    def test_selects_semantic_date_column_and_excludes_feedback(self):
        frame = pd.DataFrame(
            {
                "tweet_text": ["2026-07-19", "2026-07-20"],
                "rating": [1, 2],
                "created_at": [
                    "2026-07-19T08:00:00Z",
                    "2026-07-20T08:00:00Z",
                ],
            }
        )

        result = find_date_column(frame, excluded_column="tweet_text")

        self.assertEqual(result, "created_at")

    def test_rejects_mostly_invalid_date_values(self):
        frame = pd.DataFrame(
            {
                "date": ["not a date", "also invalid", "2026-07-20"],
                "runtime": [10, 20, 30],
                "text": ["a", "b", "c"],
            }
        )

        self.assertIsNone(find_date_column(frame, excluded_column="text"))


if __name__ == "__main__":
    unittest.main()
