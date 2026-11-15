from plotting.bar import plot_bar

CSV_PATH = "../output/summary/test_accuracy.csv"
OUT_DIR = "../output/summary/figures"


def main():
    plot_bar(
        csv_path=CSV_PATH,
        x="model",
        y="openml_4541_diabetes130us",
        xlabel="Model",
        ylabel="Accuracy",
        ylim=(0, 0.6),
        out_dir=OUT_DIR,
        output_name="model_accuracy_comparison",
    )


if __name__ == "__main__":
    main()