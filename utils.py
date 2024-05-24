import gensim
from gensim.corpora import WikiCorpus
import json
import requests
import os
from tqdm import tqdm
import multiprocessing

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='gensim')


def main():
    # Download the Wikipedia dump
    url = 'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2'
    dump_file = 'enwiki-latest-pages-articles.xml.bz2'

    if not os.path.exists(dump_file):
        print("Downloading Wikipedia dump...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
        with open(dump_file, 'wb') as f:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)
        progress_bar.close()
        print("Download complete.")

    if os.path.exists(dump_file):
        print("Skipping download of Wikipedia dump...")

    # Process the Wikipedia dump and generate JSON output
    json_file = 'enwiki-latest.json.gz'

    if not os.path.exists(json_file):
        print("Processing Wikipedia dump and generating JSON output...")
        wiki_corpus = WikiCorpus(dump_file)
        print("Started processing with gensim...")

        with gensim.utils.open(json_file, 'wb') as f:
            article_count = 0
            for article in wiki_corpus.get_texts():
                json_data = json.dumps({'text': ' '.join(article)}).encode('utf-8')
                f.write(json_data + b'\n')
                article_count += 1

                # Print a message every 1000 articles to indicate progress
                if article_count % 1000 == 0:
                    print(f"Processed {article_count} articles...")

        print("Processing complete.")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
