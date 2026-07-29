import pandas as pd


def binarize_all_features(df, max_unique_values=5):
    for col in df.columns:
        num_values = len(df[col].unique())

        if num_values == 2:
            # Ensure binary columns are mapped to 0 and 1
            unique_vals = sorted(df[col].unique())  # Sort values
            df[col] = df[col].map({unique_vals[0]: 0, unique_vals[1]: 1})
            continue

        if num_values > max_unique_values:
            df = df.drop(col, axis=1)
            continue

        dummies = pd.get_dummies(df[col], prefix=col, prefix_sep='=')
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(col, axis=1)

    return df
