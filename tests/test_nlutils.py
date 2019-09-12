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


if __name__ == "__main__":
    t = TestGetVocab()
    t.test_init()
    t.test_langs()