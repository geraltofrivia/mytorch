from src.mytorch import nlutils as nlu
from src.mytorch.utils.goodies import UnknownSpacyLang

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

    def test_simple_text(self, cores=1):
        """ Test by giving it few hardcoded sentences, and see if all the items are properly tokenized """
        input_sentences = {
            "train": [
                "He is just a little boy from a poor family",
                "Spare his life from this monstrosity"
            ],
            "valid": [
                "He is just a little boy from a poor family",
                "Spare his life from this monstrosity"
            ]
        }
        expected_output = {
            "train": [
                ["He", "is", "just", "a", "little", "boy", "from", "a", "poor", "family"],
                ["Spare", "his", "life", "from", "this", "monstrosity"]
            ],
            "valid": [
                ["He", "is", "just", "a", "little", "boy", "from", "a", "poor", "family"],
                ["Spare", "his", "life", "from", "this", "monstrosity"]
            ]
        }

        actual_output = nlu.preproc(data=input_sentences, language='en', n_threads=cores)

        assert expected_output.keys() == actual_output.keys(), "The split labels are not retained."
        for key in expected_output.keys():
            for i_seq, (expected_seq, actual_seq) in enumerate(zip(expected_output[key], actual_output[key])):
                for i_tok, (expected_tok, actual_tok) in enumerate(zip(expected_seq, actual_seq)):
                    assert expected_tok == actual_tok, f"Token {i_tok} in Seq {i_seq} differs. " \
                                                       f"Expected {expected_tok}. Got {actual_tok}"

    def test_foreign_text(self):
        """ Try with German, Spanish text. """
        ...

    def test_parallel(self):
        """ Increase the threadcount, and see if the code breaks"""
        try:
            self.test_simple_text(cores=6)
        except AssertionError:
            ...
        except Exception as e:
            raise AssertionError("Simple Test breaks when thread count is increased (Or it was broken from the getgo).")

    def test_spacy_vocab_lang(self):
        """ When giving a full vocab object in lang """
        ...


if __name__ == "__main__":
    t = TestPreProc()
    t.test_simple_text()