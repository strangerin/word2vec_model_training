import gensim
import logging
from tqdm import tqdm
import gzip
import json
import multiprocessing

# Set up logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Load the preprocessed data with progress bar
json_file = 'enwiki-latest.json.gz'
wiki_texts = []

with gzip.open(json_file, 'rt', encoding='utf-8') as f:
    total_lines = sum(1 for _ in f)

logging.info("Data processing started")
with gzip.open(json_file, 'rt', encoding='utf-8') as f:
    with tqdm(total=total_lines, unit='lines') as pbar:
        for line in f:
            article = json.loads(line)
            wiki_texts.append(gensim.utils.simple_preprocess(article['text']))
            pbar.update(1)

# Create the Word2Vec model
cores = multiprocessing.cpu_count()
logging.info(f"Number of available cores: {cores}")
model = gensim.models.Word2Vec(
    vector_size=100,
    window=5,
    min_count=5,
    workers=cores,
    epochs=5
)

# Build the vocabulary
model.build_vocab(wiki_texts)

# Train the model with progress bar
total_examples = model.corpus_count

with tqdm(total=total_examples, unit='corpus') as pbar:
    model.train(wiki_texts, total_examples=total_examples, epochs=model.epochs, callbacks=[pbar.update])

# Save the trained model
model.save('wikipedia_word2vec.model')
