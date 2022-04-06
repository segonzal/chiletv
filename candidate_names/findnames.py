import re
import csv
import json
from pathlib import Path

import argh
import spacy

# python -m spacy download es_core_news_lg


REGEX_NEWLINE = re.compile(r'\s*\n\s*')
REGEX_SPACES = re.compile(r'[^\S\r\n][^\S\r\n]+')


def clean_string(text):
    text = REGEX_NEWLINE.sub('\n', text)
    text = REGEX_SPACES.sub(' ', text)
    return text


def get_news(root: Path):
    for json_file in root.glob('*.json'):
        with json_file.open('r', encoding='utf8') as fp:
            for line in fp:
                obj = json.loads(line)
                text = '\n'.join([obj[k] for k in ['title', 'description', 'content']])
                text = clean_string(text)
            yield text


@argh.arg('root', help='Storage folder of the crawled news.')
def main(root: str):
    root = Path(root)
    model_name = 'es_core_news_lg'

    if not spacy.util.is_package(model_name):
        spacy.cli.download(model_name)

    nlp = spacy.load(model_name, disable=['tok2vec',
                                          'tagger',
                                          'parser',
                                          'attribute_ruler',
                                          'lemmatizer'])

    people = {}
    for document_number, text in enumerate(get_news(root)):
        doc = nlp(text)
        for entity in doc.ents:
            if entity.label_ == 'PER' and ' ' in entity.text:
                d = people.setdefault(entity.text, {})
                d[document_number] = d.get(document_number, 0) + 1

    with (root / 'documents.csv').open('w', encoding='utf8', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['name', 'num_docs', 'sum_values'])
        writer.writeheader()
        for name, docs in people.items():
            writer.writerow({
                'name': name,
                'num_docs': len(docs),
                'sum_values': sum(docs.values()),
            })


if __name__ == '__main__':
    argh.dispatch_command(main)
