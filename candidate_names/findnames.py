import re
import csv
import json
import spacy
import numpy as np
from pathlib import Path
from spacy.kb import KnowledgeBase
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


def main():
    root = Path('news')
    nlp = spacy.load('es_core_news_lg', disable=['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer'])

    people = {}
    for document_number, text in enumerate(get_news(root)):
        doc = nlp(text)
        for entity in doc.ents:
            if entity.label_ == 'PER' and ' ' in entity.text:
                d = people.setdefault(entity.text, {})
                d[document_number] = d.get(document_number, 0) + 1

    with (root / 'documents.csv').open('w', encoding='utf8') as csvfile:
        for name, docs in people.items():
            csvfile.write(f"{name},{len(docs)},{sum(docs.values())}\n")



if __name__ == '__main__':
    main()
