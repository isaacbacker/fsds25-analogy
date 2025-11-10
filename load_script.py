import ssl
import certifi
import os

# Configure SSL to use certifi's certificate bundle
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

# Create SSL context with certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
ssl._create_default_https_context = lambda: ssl_context

import gensim.downloader as api
from src.analogy_tests import (
    run_analogy_test_suite,
    print_test_summary,
)

import pandas as pd
from src.models import ModelManager
from src.analogy_tests import test_analogy

# === CONFIG ===
csv_path = 'data/migration_sample_analogies.csv'
output_path = csv_path  # overwrite same file; change if you prefer to save a copy
top_n = 10
search_space = 50000

# === Load CSV ===
df = pd.read_csv(csv_path)
print(f"Loaded CSV with {len(df)} rows.")

# Detect column names for the analogy
colA, colB, colC, colD = "word1", "word2", "word3", "word4 (target)"


#Twitter model (uncommment below to run)
#model = api.load("glove-twitter-200")

#fasttext model (uncomment below to run)
model = api.load("fasttext-wiki-news-subwords-300")

#Original word2vec model (uncomment below to run)
#model = api.load("word2vec-google-news-300")


# === Process each row ===
for i, row in df.iterrows():
    a, b, c, target = str(row[colA]), str(row[colB]), str(row[colC]), str(row[colD])
    print(f"\n[{i+1}/{len(df)}] {a}:{b}::{c}:{target}")

    try:
        result = test_analogy(model, a, b, c, target, top_n=top_n, search_space=search_space)

        # Handle both possible return types: (neighbors, rank, pred) or (neighbors, rank)
        pred, rank = None, None
        if result is None:
            pass
        elif isinstance(result, (list, tuple)):
            if len(result) == 3:
                neighbors, rank, pred = result
            elif len(result) == 2:
                neighbors, rank = result
                pred = neighbors[0][0] if neighbors else None
        else:
            pass

        df.at[i, "pred_fasttextwiki"] = pred if pred else ""
        df.at[i, "targetrank_fasttextwiki"] = rank if rank else None

    except Exception as e:
        print(f"Error on row {i}: {e}")
        df.at[i, "pred_fasttextwiki"] = ""
        df.at[i, "targetrank_fasttextwiki"] = None

df.to_csv(output_path, index=False)




#results = run_analogy_test_suite(model, csv_path='data/analogies.csv')
#print_test_summary(results)
#print(results)

