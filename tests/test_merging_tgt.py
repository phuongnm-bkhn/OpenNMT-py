from onmt.translate import Translator

print(' '.join(Translator.merge_sentence_label([
    "O", "B-g", "I-g"
], [
    "a", "b", "c"
], ["_a", "g"])))
