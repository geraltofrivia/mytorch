from mytorch import nlutils as nlu
from mytorch.utils.goodies import UnknownSpacyLang

class TestGetVocab:
    """ Test the internal get vocab function. """

    def test_init(self):
        try:
            _ = nlu._get_spacy_vocab_('en')
        except Exception as e:
            raise AssertionError(f"The function does not work. Got exception: {e}")

    def test_langs(self):
        """ Test if it works with the languages I've hardcoded already. """
        langs = {
            'en': ['en', 'english'],
            'es': ['es', 'espaneol', 'espanhol', 'spanish'],
            'de': ['de', 'german', 'deutsch']
        }

        for spacy_lang_name, lang_names in langs.items():
            for lang_name in lang_names:
                try:
                    vocab = nlu._get_spacy_vocab_(lang_name)
                    assert vocab.lang == spacy_lang_name, f"Wrong vocabulary returned: {vocab.lang}. " \
                                                          f"Expected {spacy_lang_name}."
                except UnknownSpacyLang:
                    raise AssertionError(f"Expected Language {lang_name} to be known.")


class TestPreProc:
    """ Tests for the preproc method which can tokenize stuff """

    def test_init(self):
        ...

    def test_simple_text(self):
        """ Test by giving it few hardcoded sentences, and see if all the items are properly tokenized """
        ...

    def test_foreign_text(self):
        """ Try with German, Spanish text. """
        ...

    def test_parallel(self):
        """ Increase the threadcount, and see if you get a time improvement """
        ...

    def test_spacy_vocab_lang(self):
        """ When giving a full vocab object in lang """
        ...


if __name__ == "__main__":
    t = TestGetVocab()
    t.test_init()
    t.test_langs()