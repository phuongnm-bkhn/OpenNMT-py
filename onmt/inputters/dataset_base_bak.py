# coding: utf-8
import logging
import re
from itertools import chain, starmap
from collections import Counter

import torch
from torchtext.data import Dataset as TorchtextDataset
from torchtext.data import Example
from torchtext.vocab import Vocab


def _join_dicts(*args):
    """
    Args:
        dictionaries with disjoint keys.

    Returns:
        a single dictionary that has the union of these keys.
    """

    return dict(chain(*[d.items() for d in args]))


def _dynamic_dict(example, src_field, tgt_field):
    """Create copy-vocab and numericalize with it.

    In-place adds ``"src_map"`` to ``example``. That is the copy-vocab
    numericalization of the tokenized ``example["src"]``. If ``example``
    has a ``"tgt"`` key, adds ``"alignment"`` to example. That is the
    copy-vocab numericalization of the tokenized ``example["tgt"]``. The
    alignment has an initial and final UNK token to match the BOS and EOS
    tokens.

    Args:
        example (dict): An example dictionary with a ``"src"`` key and
            maybe a ``"tgt"`` key. (This argument changes in place!)
        src_field (torchtext.data.Field): Field object.
        tgt_field (torchtext.data.Field): Field object.

    Returns:
        torchtext.data.Vocab and ``example``, changed as described.
    """

    src = src_field.tokenize(example["src"])
    # make a small vocab containing just the tokens in the source sequence
    unk = src_field.unk_token
    pad = src_field.pad_token
    src_ex_vocab = Vocab(Counter(src), specials=[unk, pad])
    unk_idx = src_ex_vocab.stoi[unk]
    # Map source tokens to indices in the dynamic dict.
    src_map = torch.LongTensor([src_ex_vocab.stoi[w] for w in src])
    example["src_map"] = src_map

    if "tgt" in example:
        tgt = tgt_field.tokenize(example["tgt"])
        mask = torch.LongTensor(
            [unk_idx] + [src_ex_vocab.stoi[w] for w in tgt] + [unk_idx])
        example["alignment"] = mask
    return src_ex_vocab, example


class Dataset(TorchtextDataset):
    """Contain data and process it.

    A dataset is an object that accepts sequences of raw data (sentence pairs
    in the case of machine translation) and fields which describe how this
    raw data should be processed to produce tensors. When a dataset is
    instantiated, it applies the fields' preprocessing pipeline (but not
    the bit that numericalizes it or turns it into batch tensors) to the raw
    data, producing a list of :class:`torchtext.data.Example` objects.
    torchtext's iterators then know how to use these examples to make batches.

    Args:
        fields (dict[str, Field]): a dict with the structure
            returned by :func:`onmt.inputters.get_fields()`. Usually
            that means the dataset side, ``"src"`` or ``"tgt"``. Keys match
            the keys of items yielded by the ``readers``, while values
            are lists of (name, Field) pairs. An attribute with this
            name will be created for each :class:`torchtext.data.Example`
            object and its value will be the result of applying the Field
            to the data that matches the key. The advantage of having
            sequences of fields for each piece of raw input is that it allows
            the dataset to store multiple "views" of each input, which allows
            for easy implementation of token-level features, mixed word-
            and character-level models, and so on. (See also
            :class:`onmt.inputters.TextMultiField`.)
        readers (Iterable[onmt.inputters.DataReaderBase]): Reader objects
            for disk-to-dict. The yielded dicts are then processed
            according to ``fields``.
        data (Iterable[Tuple[str, Any]]): (name, ``data_arg``) pairs
            where ``data_arg`` is passed to the ``read()`` method of the
            reader in ``readers`` at that position. (See the reader object for
            details on the ``Any`` type.)
        dirs (Iterable[str or NoneType]): A list of directories where
            data is contained. See the reader object for more details.
        sort_key (Callable[[torchtext.data.Example], Any]): A function
            for determining the value on which data is sorted (i.e. length).
        filter_pred (Callable[[torchtext.data.Example], bool]): A function
            that accepts Example objects and returns a boolean value
            indicating whether to include that example in the dataset.

    Attributes:
        src_vocabs (List[torchtext.data.Vocab]): Used with dynamic dict/copy
            attention. There is a very short vocab for each src example.
            It contains just the source words, e.g. so that the generator can
            predict to copy them.
    """

    def __init__(self, fields, readers, data, dirs, sort_key,
                 filter_pred=None):
        self.sort_key = sort_key
        can_copy = 'src_map' in fields and 'alignment' in fields

        count_sample = len(data[0][1])
        transformed_data = []
        for i in range(count_sample):
            src_str, tgt_str = data[0][1][i], data[1][1][i]
            encode_words, decode_words, decode_transformed, encode_labels = \
                Dataset.matching_enc_label(src_str.decode("utf-8") , tgt_str.decode("utf-8") )
            data[1][1][i] = (" ".join(decode_transformed)).encode('utf-8')
            data[0][1][i] = (" ".join(encode_words)).encode('utf-8')
            transformed_data.append((encode_words, decode_words, decode_transformed, encode_labels))

        read_iters = [r.read(dat[1], dat[0], dir_) for r, dat, dir_
                      in zip(readers, data, dirs)]

        # self.src_vocabs is used in collapse_copy_scores and Translator.py
        self.src_vocabs = []
        examples = []
        for ex_dict in starmap(_join_dicts, zip(*read_iters)):
            if can_copy:
                src_field = fields['src']
                tgt_field = fields['tgt']
                # this assumes src_field and tgt_field are both text
                src_ex_vocab, ex_dict = _dynamic_dict(
                    ex_dict, src_field.base_field, tgt_field.base_field)
                self.src_vocabs.append(src_ex_vocab)
            ex_fields = {k: [(k, v)] for k, v in fields.items() if
                         k in ex_dict}
            ex = Example.fromdict(ex_dict, ex_fields)
            examples.append(ex)

        # fields needs to have only keys that examples have as attrs
        org_fields = fields
        fields = []
        for _, nf_list in ex_fields.items():
            assert len(nf_list) == 1
            fields.append(nf_list[0])
        for k in org_fields:
            if k not in ex_fields:
                fields.append((k, org_fields[k]))

        for sample in examples:
            setattr(sample, 'src_label', [transformed_data[sample.indices][3]])

        super(Dataset, self).__init__(examples, fields, filter_pred)

    def __getattr__(self, attr):
        # avoid infinite recursion when fields isn't defined
        if 'fields' not in vars(self):
            raise AttributeError
        if attr in self.fields:
            return (getattr(x, attr) for x in self.examples)
        else:
            raise AttributeError

    def save(self, path, remove_fields=True):
        if remove_fields:
            self.fields = []
        torch.save(self, path)

    @staticmethod
    def matching_enc_label(encode_str: str, decode_str: str):
        encode_str = encode_str.strip()
        decode_str = decode_str.strip()

        # - case: (us|united states|america) in encode and (usa) in decode
        encode_str = Dataset.abbreviate_recover(encode_str)

        encode_words = encode_str.split()
        encode_labels = ["O"] * len(encode_words)
        decode_words = decode_str.split()
        w_intersection = set(encode_words).intersection(set(decode_words))

        import copy
        w_intersection2 = copy.deepcopy(w_intersection)
        for w in w_intersection2:
            if not re.match(r".*\d$", w):
                w_intersection.remove(w)

        last_idx = -1
        elements = []
        for i, w in enumerate(encode_words + ["#<end>"]):
            if w in w_intersection:
                if i - last_idx == 1 and last_idx != -1 and (elements[-1] + " " + w) in decode_str:
                    elements[-1] += " " + w
                    encode_labels[i] = "I-arg_{}:lb".format(len(elements))
                else:
                    elements.append(w)
                    encode_labels[i] = "B-arg_{}:lb".format(len(elements))
                last_idx = i
            else:
                last_idx = -1
        for i, e_val in enumerate(elements):
            term_search = re.compile("( |^){}( |$)".format(e_val.replace("+","\+")))
            if term_search.search(decode_str) is not None:
                decode_str = term_search.sub('\\1arg_{}:lb\\2'.format(i + 1), decode_str, count=1)
            else:
                logging.warning("- [W] {} not exist in decode_str {}".format(e_val, decode_str))

        # valid check with simple rules for
        # if len(elements) == 0 and "id" in decode_str:
        #     logging.warning("- [W] id had not replaced exist in \n {} \n {}".format(decode_str, encode_str))

        decode_transformed = decode_str.split()
        return encode_words, decode_words, decode_transformed, encode_labels

    @staticmethod
    def abbreviate_recover(input_str):
        abbr_vocab = {
            "us": "usa",
            "united states": "usa",
            "america": "usa",
        }
        for k, v in abbr_vocab.items():
            input_str = re.sub("( |^){}( |$)".format(k), '\\1{}\\2'.format(v), input_str)
        return input_str