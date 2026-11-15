import os
from plotting.line import plot_line

# CSV_PATH = "../output/mlp/training_history.csv"
# OUT_DIR = "../output/mlp/figures"
CSV_PATH = "../output/eapcr/training_history.csv"
OUT_DIR = "../output/eapcr/figures"

def main():
    #两条线都不要点
    plot_line(
        csv_path=CSV_PATH,
        x="epoch",
        ys=["train_accuracy", "test_accuracy"],
        labels=["Train", "Test"],
        markers=[None, None],
        xlabel="Epoch",
        ylabel="Accuracy",
        out_dir=OUT_DIR,
        output_name="accuracy_curve_no_markers",
        ylim=(0, 1.03),
    )


    #“Train 没点，Test 有点”
    # plot_line(
        # csv_path=CSV_PATH,
        # x="epoch",
        # ys=["train_accuracy", "test_accuracy"],
        # labels=["Train", "Test"],
        # markers=[None, "o"],
        # xlabel="Epoch",
        # ylabel="Accuracy",
        # out_dir=OUT_DIR,
        # output_name="accuracy_curve",
        # ylim=(0, 1.03),
    # )


    #所有线都有点
    # plot_line(
        # csv_path=CSV_PATH,
        # x="epoch",
        # ys=["train_accuracy", "test_accuracy"],
        # labels=["Train", "Test"],
        # markers=["o", "o"],
        # xlabel="Epoch",
        # ylabel="Accuracy",
        # out_dir=OUT_DIR,
        # output_name="accuracy_curve_with_markers",
        # ylim=(0, 1.03),
    # )




if __name__ == "__main__":
    main()