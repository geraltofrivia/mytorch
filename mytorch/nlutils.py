"""
    Everything explicitly pertaining to NL things come here.

    **Tokenization**
    1. Input
"""
import spacy
from spacy.tokenizer import Tokenizer

# Local Imports
from .utils.goodies import *

def _get_spacy_vocab_(lang: str) -> spacy.vocab.Vocab:
    """
        Returns a spacy Vocab object to be used in tokenizer. Because multilinguality!
        TODO: Hard code languages!
    :param lang: a string of languages
    :return: a spacy vocab object
    """

    if lang.lower() in ['en', 'english']:
        from spacy.lang.en import English as Lang
    elif lang.lower() in ['es', 'español', 'espanyol', 'espanhol', 'espaneol', 'spanish']:
        from spacy.lang.es import Spanish as Lang
    elif lang.lower() in ['de', 'german', 'deutsch']:
        from spacy.lang.de import German as Lang
    else:
        raise UnknownSpacyLang(f"The language {lang} is either not hardcoded in yet, or not available in spaCy. Sorry.")

    nlp = Lang()
    return nlp.vocab


def preproc(data: Dict[str, List[str]], language: Union[spacy.vocab.Vocab, str] = 'en',
            n_threads: int = 1, vocabularize: bool = False) \
        -> (Dict[str, List[List]], Optional[List[str]], Optional[Dict[str, int]]):
    """
        A slightly abstract method to

        |-> tokenize text
        |-> idfy/vocabularize/onehot-indexify text

        **Example**
        tok_txt = preproc({'train': tr_txt, 'test': ts_txt, 'valid': vl_txt}, 'en')
        tok_txt, stoi = preproc( ... , vocabularize = True)

    :param data: A dict of {'any random tag': list/iter of data}
    :param language: A string which decides what language does the next belong to.
    :param n_threads: 1
    :param vocabularize: a flag which if turned on, will also convert text to ID and return relevant mappings
    :return: if vocabularize:
                -> Dict of original keys and values where values are no longer a list of strings but a list of list of ints
                -> A list of unique tokens
                -> A corresponding Dict of {str:int}
            else:
                -> Dict of original keys and values where values are no longer a list of strings but a list of list of strs
    """

    stoi, tok_data = {}, {k: [] for k in data.keys()}

    # Get spacy Vocab
    vocab = _get_spacy_vocab_(language) if type(language) is str else language
    tokenizer = Tokenizer(vocab)

    for key, value in data.items():
        for doc in tokenizer.pipe(value, batch_size=10000, n_threads=5):
            if vocabularize:
                tok_data[key].append([stoi.setdefault(tok.text, len(stoi)) for tok in doc])
            else:
                tok_data[key].append([tok.text for tok in doc])

    if vocabularize:
        return tok_data
    else:
        return tok_data, stoi
