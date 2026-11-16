# Atlas — Data Science Research Agent

An interactive research workspace for **exploratory data analysis, visual storytelling, baseline machine learning, and optional AI interpretation**.

[![Python CI](https://github.com/vikas56655/DS-Research-Agent/actions/workflows/ci.yml/badge.svg)](https://github.com/vikas56655/DS-Research-Agent/actions/workflows/ci.yml)

## What you can do

- Explore a deterministic **synthetic** customer-retention dataset immediately, or upload your own UTF-8 CSV (15 MB limit).
- Inspect data quality: missing values, duplicates, column types and a searchable table.
- Explore **seven interactive visualizations**: missingness, data types, histograms with boxplots, category frequency, correlation heatmap, scatterplots, and model diagnostics.
- Train reproducible **classification** (logistic regression) or **regression** (ridge regression) baselines with a fixed holdout split.
- Download a Markdown research brief.
- Optionally request OpenAI-generated commentary based on **aggregate statistics only**, not raw records.

## Screens and visualizations

| Workspace | Chart | Question answered |
| --- | --- | --- |
| Overview | Missingness bar chart | Where are the data-quality gaps? |
| Overview | Feature-type donut | What kinds of features exist? |
| Distributions | Histogram + boxplot | How is a numerical variable distributed? |
| Distributions | Category bar chart | Which groups dominate? |
| Relationships | Correlation heatmap | Which numerical features co-vary? |
| Relationships | Scatter plot | How do two features relate? |
| Model lab | Confusion matrix / actual vs predicted | How does the baseline perform? |

## Quick start

Python 3.11 or 3.12 recommended.

~~~bash
git clone https://github.com/vikas56655/DS-Research-Agent.git
cd DS-Research-Agent
python -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run app.py
~~~

Open the local URL shown by Streamlit (normally http://localhost:8501).

### Optional AI interpretation

Enter an OpenAI API key in the Streamlit sidebar, or create a local environment variable named OPENAI_API_KEY. Never commit API keys or personal datasets. AI interpretation uses the OpenAI API and may incur charges. The core dashboard works without a key.

## Validation

~~~bash
python -m compileall -q app.py main.py research
python -m pytest -q
python main.py
~~~

GitHub Actions tests Python 3.11 and 3.12 on pushes and pull requests.

## Project layout

~~~text
app.py                  Streamlit dashboard and Plotly charts
main.py                 CLI smoke test
research/analysis.py    CSV parsing, profiling and ML baselines
research/ai.py          Optional read-only OpenAI interpretation
agents/                 Legacy interface note
tests/                  Automated unit tests
.github/workflows/      Continuous integration
.streamlit/             Dashboard theme
~~~

## Methodology and limitations

The built-in data are generated, not scraped or live. Uploaded data remain in application memory for local processing; AI requests transmit only aggregate statistics and the research question. No arbitrary LLM-generated code is executed.

The random 75/25 train/test split is suitable for a simple demonstration, but **not** for time series, grouped data or datasets with target leakage. Baselines are not tuned or calibrated. Correlation does not imply causation. Classification requires 2–20 classes; free-text/high-cardinality predictors are excluded. Do not upload sensitive information to third-party hosted instances without reviewing their privacy and retention practices.

## Deployment

Deploy on Streamlit Community Cloud: select this repository, branch main, and entry point app.py. Configure secrets privately if you want AI interpretation. Deployment is optional; this repository does not imply a live hosted service.

## License

MIT.
