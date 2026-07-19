import unittest

import pandas as pd

from data_adapter import normalize_external_reviews


class NormalizeExternalReviewsTest(unittest.TestCase):
    def test_maps_xquik_export_columns(self):
        df = normalize_external_reviews(
            pd.DataFrame(
                [
                    {
                        "tweet_text": "Layanan cepat dan jelas",
                        "sentiment_label": "positive",
                        "created_at": "2026-07-05T12:00:00Z",
                    }
                ]
            )
        )

        self.assertEqual(df.loc[0, "teks_ulasan"], "Layanan cepat dan jelas")
        self.assertEqual(df.loc[0, "sentimen_prediksi"], "Positif")
        self.assertEqual(int(df.loc[0, "rating"]), 5)
        self.assertEqual(int(df.loc[0, "bulan"]), 7)

    def test_maps_current_xquik_tweet_date_column(self):
        df = normalize_external_reviews(
            pd.DataFrame(
                [
                    {
                        "text": "Data Xquik terbaru",
                        "createdAt": "2026-09-05T12:00:00Z",
                    }
                ]
            )
        )

        self.assertEqual(df.loc[0, "teks_ulasan"], "Data Xquik terbaru")
        self.assertEqual(df.loc[0, "sentimen_prediksi"], "Netral")
        self.assertEqual(int(df.loc[0, "bulan"]), 9)

    def test_rejects_missing_text_column(self):
        with self.assertRaisesRegex(ValueError, "kolom teks"):
            normalize_external_reviews(pd.DataFrame([{"score": 1}]))

    def test_accepts_trimmed_case_insensitive_headers_and_numeric_month(self):
        df = normalize_external_reviews(
            pd.DataFrame(
                [
                    {
                        " Text ": "  Prosesnya terlalu lama  ",
                        " SENTIMENT ": "NEGATIVE",
                        " Month ": 7,
                    },
                    {
                        " Text ": "   ",
                        " SENTIMENT ": "positive",
                        " Month ": 8,
                    },
                ]
            )
        )

        self.assertEqual(len(df), 1)
        self.assertEqual(df.loc[0, "teks_ulasan"], "Prosesnya terlalu lama")
        self.assertEqual(df.loc[0, "sentimen_prediksi"], "Negatif")
        self.assertEqual(int(df.loc[0, "rating"]), 1)
        self.assertEqual(int(df.loc[0, "bulan"]), 7)

    def test_rejects_missing_time_column(self):
        with self.assertRaisesRegex(ValueError, "kolom waktu"):
            normalize_external_reviews(pd.DataFrame([{"text": "Bagus"}]))

    def test_rejects_invalid_date(self):
        with self.assertRaisesRegex(ValueError, "tanggal"):
            normalize_external_reviews(
                pd.DataFrame([{"text": "Bagus", "created_at": "bukan tanggal"}])
            )

    def test_rejects_out_of_range_month(self):
        with self.assertRaisesRegex(ValueError, "1 sampai 12"):
            normalize_external_reviews(pd.DataFrame([{"text": "Bagus", "bulan": 13}]))

    def test_defaults_unknown_sentiment_to_neutral(self):
        df = normalize_external_reviews(
            pd.DataFrame([{"text": "Biasa saja", "sentiment": "mixed", "date": "2026-02-01"}])
        )

        self.assertEqual(df.loc[0, "sentimen_prediksi"], "Netral")
        self.assertEqual(int(df.loc[0, "rating"]), 3)
        self.assertEqual(int(df.loc[0, "bulan"]), 2)


if __name__ == "__main__":
    unittest.main()
