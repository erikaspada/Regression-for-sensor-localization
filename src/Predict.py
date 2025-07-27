import pandas as pd

def generate_predictions(model, df_eval, output_path):
    X_eval = df_eval.drop(['Id'], axis=1)
    ids = df_eval['Id']

    y_pred = model.predict(X_eval)
    formatted_predictions = [f"{x}|{y}" for x, y in y_pred]

    df_output = pd.DataFrame(formatted_predictions, index=ids)
    df_output.index.name = "Id"
    df_output.columns = ["Predicted"]
    df_output.to_csv(output_path)
