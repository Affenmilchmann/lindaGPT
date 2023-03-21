from pymorphy2 import MorphAnalyzer
from re import sub as re_sub, match as re_match
from tqdm import tqdm

import logging
logging.basicConfig(format='Tokenizer:\t%(asctime)s %(message)s', datefmt='[%H:%M:%S]', level=logging.INFO)

def is_a_word(token: str) -> bool:
    return re_match(r'^[а-я]+$', token)

def remove_low_frequent(tokens: list, threshold: int = 10):
    logging.info("Counting frequencies")
    freq_map = {}
    for tkn in tqdm(tokens):
        if tkn in freq_map: freq_map[tkn] += 1
        else: freq_map[tkn] = 1

    logging.info(f"Removing with freq < {threshold}")
    return [ t for t in tokens if freq_map[t] >= threshold ]

def tokenize_text(text: str) -> list:
    logging.info("Tokenizing")
    morph = MorphAnalyzer()
    
    text = text.lower()
    text = re_sub(r'([^а-яА-Яё ])', ' \g<1> ', text)
    text = re_sub(r' +', ' ', text)

    tokens = text.split(' ')
    tokens = remove_low_frequent(tokens)

    """ for tkn in text.split():
        if is_a_word(tkn):
            parsed_tkn = morph.parse(tkn)
            parsed_tkn
        else:
            tokens.append(tkn) """

    return tokens

if __name__ == "__main__":
    with open('data/anekdots.txt', 'r', encoding='utf-8') as f_in:
        with open('tmp.txt', 'w', encoding='utf-8') as f_out:
            f_out.write(tokenize_text(f_in.read()))
