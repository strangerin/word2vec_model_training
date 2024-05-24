import gensim
import logging
from tqdm import tqdm
import gzip
import json

# Set up logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Load the preprocessed data
json_file = 'enwiki-latest.json.gz'
with gzip.open(json_file, 'rt', encoding='utf-8') as f:
    wiki_texts = []
    for line in f:
        article = json.loads(line)
        wiki_texts.append(article['text'])

# Create the Word2Vec model
model = gensim.models.Word2Vec(
    wiki_texts,
    vector_size=100,
    window=5,
    min_count=5,
    workers=4
)

# Train the model with progress bar
epochs = 5
total_examples = len(wiki_texts)

with tqdm(total=total_examples * epochs) as pbar:
    for epoch in range(epochs):
        model.train(wiki_texts, total_examples=total_examples, epochs=1)
        pbar.update(total_examples)

# Save the trained model
model.save('wikipedia_word2vec.model')
