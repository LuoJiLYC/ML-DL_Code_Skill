from plotting.line import plot_line_by_row_models

CSV_PATH = "../output/summary/acc_compare.csv"
OUT_DIR = "../output/summary/fig"

def main():
    plot_line_by_row_models(
        csv_path=CSV_PATH,
        out_dir=OUT_DIR,
        output_name="accuracy_vs_datasize",
        model_col="model",
        xlabel="Training Set Size",
        ylabel="Accuracy",
        ylim=(0.5, 0.6),
    )


if __name__ == "__main__":
    main()