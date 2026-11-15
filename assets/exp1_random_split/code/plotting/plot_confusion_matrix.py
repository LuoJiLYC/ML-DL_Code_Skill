from plotting.confusion_matrix import plot_confusion_matrix

# JSON_PATH = "../output/catboost/result.json"
# OUT_DIR = "../output/catboost/figures"
# JSON_PATH = "../output/mlp/result.json"
# OUT_DIR = "../output/mlp/figures"
JSON_PATH = "../output/eapcr/result.json"
OUT_DIR = "../output/eapcr/figures"
# JSON_PATH = "../output/tabpfn/result.json"
# OUT_DIR = "../output/tabpfn/figures"


def main():
    plot_confusion_matrix(
        json_path=JSON_PATH,
        labels=["0", "1", "2"],
        out_dir=OUT_DIR,
        output_name="confusion_matrix",
    )


if __name__ == "__main__":
    main()