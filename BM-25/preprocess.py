import os
import pickle

import pyarrow as pa
import pandas as pd

def make_string_to_list(corpus_list: list) -> list:
    splitted_corpus_list = list(map(lambda doc: doc.split(" "),corpus_list))
    return splitted_corpus_list

if __name__ == "__main__":
    # Load Dataset to Prepreocess
    origin_path = "/Users/hwanghyejeong/Downloads"
    parquet_path = os.path.join(origin_path,"data_preprocessed.parquet")

    df = pd.read_parquet(parquet_path)
    print(f"Before Preprocess {len(df)} rows and {df.columns} columns")

    # Preprocess
    df["bm25_input"] = df["text"].apply(lambda text: text.split(" "))
    print(f"After Preprocess {len(df)} rows and {df.columns} columns")

    # Save Preprocessed Dataset to Parquet
    new_parquet_path = os.path.join(origin_path,"bm25input_added_data.parquet")
    df.to_parquet(new_parquet_path, index=False)
    del df

    