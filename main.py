from src.preprocessing import preprocess_data
from src.model import train_model
from src.predict import generate_predictions
from src.visualization import spatial_distribution, dataset_balance_plot


def main():
    dev_path = "data/development.csv"
    eval_path = "data/evaluation.csv"
    output_path = "output/output.csv"

    print("\n[INFO] Preprocessing data...")
    df_train, df_eval = preprocess_data(dev_path, eval_path)

    print("[INFO] Visualizing spatial distribution...")
    spatial_distribution(df_train)

    print("[INFO] Visualizing dataset balance...")
    dataset_balance_plot(df_train)

    print("[INFO] Training model...")
    model = train_model(df_train)

    print("[INFO] Generating predictions...")
    generate_predictions(model, df_eval, output_path)

    print(f"[DONE] Predictions saved to {output_path}")


if __name__ == "__main__":
    main()
