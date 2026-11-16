"""CLI smoke test. Launch the full interface with: streamlit run app.py"""
from research.analysis import baseline, sample_data


def main() -> None:
    df = sample_data()
    result = baseline(df, "churn", "classification")
    print("Atlas DS Research Agent — demo smoke test")
    print(f"Demo observations: {len(df)}")
    for metric, value in result["metrics"].items():
        print(f"{metric}: {value:.3f}")
    print("For interactive charts run: streamlit run app.py")


if __name__ == "__main__":
    main()
