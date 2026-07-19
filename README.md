# AdaKami Sentiment Dashboard

Streamlit dashboard for AdaKami review sentiment analysis across the January to
December 2025 dataset.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run dashboard.py
```

The default dataset lives in `data/ulasan_dengan_prediksi.csv`.

## Social CSV Uploads

The sidebar accepts additional Xquik or social export CSV rows and appends them
to the dashboard data before charts are calculated.

Supported text columns:

- `text`
- `tweet_text`
- `review`
- `comment`
- `content`
- `body`

Supported sentiment columns:

- `sentiment`
- `sentiment_label`
- `sentimen`
- `sentimen_prediksi`
- `label`

Supported date columns:

- `created_at`
- `createdAt`
- `timestamp`
- `date`
- `month`
- `bulan`

Column names are matched without regard to capitalization or surrounding
spaces. Rows without text are ignored. Unknown or missing sentiment values are
treated as `Netral`.

Every imported row with text must have a valid date or an integer month from 1
to 12. The dashboard rejects invalid timeline data instead of silently placing
it in January.

Xquik is an independent third-party service. Not affiliated with X Corp.
"Twitter" and "X" are trademarks of X Corp.
