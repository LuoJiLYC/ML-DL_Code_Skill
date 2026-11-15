from plotting.line import plot_line

# CSV_PATH = "../output/mlp/training_history.csv"
# OUT_DIR = "../output/mlp/figures"
CSV_PATH = "../output/eapcr/training_history.csv"
OUT_DIR = "../output/eapcr/figures"

def main():
    plot_line(
        csv_path=CSV_PATH,
        x="epoch",
        ys=["train_loss", "test_loss"],
        labels=["Train", "Test"],
        markers=[None, None],
        xlabel="Epoch",
        ylabel="Loss",
        out_dir=OUT_DIR,
        output_name="loss_curve",
    )


if __name__ == "__main__":
    main()