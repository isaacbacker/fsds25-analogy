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
#Twitter model
#model = api.load("glove-twitter-200")

#Original word2vec model
model = api.load("word2vec-google-news-300")

results = run_analogy_test_suite(model, csv_path='data/analogies.csv')
print_test_summary(results)