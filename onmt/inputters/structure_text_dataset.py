import copy
import logging
import re
from functools import partial
from typing import List

import torch
from torchtext.data import Field

from onmt.inputters.text_dataset import TextMultiField, _feature_tokenize

logger = logging.getLogger()


class DifferenceCode:
    PREDICATE = 0
    VARIABLE = 1
    CONSTANT = 2
    CHILD_COUNT = 3


class LogicElement:
    DEFAULT_RELAX_CHILD_ORDER = {"and", "or", "", "next_to"}
    DEFAULT_ALLOW_CHILD_DUPLICATION = {"and", "or", ""}

    def __init__(self, value="", child=None, relax_child_order=False, allow_child_duplication=False):
        self.child = child or []
        self.value = str(value)
        if value in LogicElement.DEFAULT_RELAX_CHILD_ORDER:
            relax_child_order = True
        self.relax_child_order = relax_child_order

        if value in LogicElement.DEFAULT_ALLOW_CHILD_DUPLICATION:
            allow_child_duplication = True
        self.allow_child_duplication = allow_child_duplication

        self.leaf_nodes = None

    def add_child(self, child):
        if isinstance(child, LogicElement):
            self.child.append(child)
        else:
            logger.warning("Can't add child that is not object LogicElement: {}" % {child})

    def is_variable_node(self, term_check=r'[\$\?][^\s\n]*'):
        if len(self.child) == 0 and re.fullmatch(term_check, self.value):
            return True
        else:
            return False

    def is_constant(self, term_check_variable=r'[\$\?][^\s\n]*'):
        if len(self.child) == 0 and not self.is_variable_node(term_check_variable):
            return True
        else:
            return False

    def is_triple(self):
        if len(self.child) > 0 and len(self.value) > 0:
            return True
        else:
            return False

    def is_leaf_node(self):
        # if node that have nephew node
        for ee in self.child:
            if isinstance(ee, LogicElement) and len(ee.child) > 0:
                return False
        return True

    def get_leaf_nodes(self):
        tmp_leaf_nodes = []
        if self.leaf_nodes is not None:
            return self.leaf_nodes
        elif self.is_leaf_node():
            tmp_leaf_nodes = [self]
        elif self.leaf_nodes is None:
            for e in self.child:
                tmp_leaf_nodes += e.get_leaf_nodes()
        self.leaf_nodes = tmp_leaf_nodes
        return tmp_leaf_nodes

    def get_path_to_leaf_nodes(self, path_to_leaf_nodes=None, cur_path=None):
        path_to_leaf_nodes = [] if path_to_leaf_nodes is None else path_to_leaf_nodes
        cur_path = [] if cur_path is None else copy.deepcopy(cur_path)
        if len(self.value) > 0:
            cur_path.append(self.value)
        if self.is_leaf_node():
            path_to_leaf_nodes.append(cur_path)
        else:
            for e in self.child:
                e.get_path_to_leaf_nodes(path_to_leaf_nodes, cur_path)
        return path_to_leaf_nodes

    def get_triple_name(self):
        tmp_triple_name = []
        if self.is_triple():
            tmp_triple_name.append(self.value)

        if len(self.child) > 0:
            for e in self.child:
                tmp_triple_name += e.get_triple_name()

        return tmp_triple_name

    def get_constant(self):
        tmp_triple_name = []
        if self.is_constant():
            return [self.value]
        for e in self.child:
            tmp_triple_name += e.get_constant()

        return tmp_triple_name

    @staticmethod
    def _collapse_list_logic(logics: List) -> List:
        len_logic = len(logics)
        mask_remove = [False] * len_logic
        for i in range(len_logic - 1):
            for j in range(i + 1, len_logic):
                if not mask_remove[j]:
                    if logics[i] == logics[j]:
                        mask_remove[j] = True
        for j in range(len_logic - 1, -1, -1):
            if mask_remove[j]:
                logics.pop(j)

        return logics

    def __eq__(self, other):
        if not isinstance(other, LogicElement) or not self.value == other.value:
            return False

        if self.allow_child_duplication and self.relax_child_order:
            self_child = copy.deepcopy(self.child)
            other_child = copy.deepcopy(other.child)

            self_child = self._collapse_list_logic(self_child)
            other_child = self._collapse_list_logic(other_child)
        else:
            self_child = self.child
            other_child = other.child

        if not len(self_child) == len(other_child):
            return False
        else:
            if not self.relax_child_order:
                for i, e in enumerate(other_child):
                    if not e == self_child[i]:
                        return False
            else:
                for i, e in enumerate(other_child):
                    j = 0
                    for _, e2 in enumerate(self_child):
                        if e == e2:
                            break
                        j += 1
                    if j == len(self_child):
                        return False
            return True

    def __str__(self):
        if len(self.child) == 0:
            return self.value
        child_str = " ".join([str(v) for v in self.child])
        if len(self.value) > 0 and len(child_str) > 0:
            return "( {} {} )".format(self.value, child_str)
        elif len(self.value) > 0:
            return "( {} )".format(self.value)
        elif len(child_str) > 0:
            return "( {} )".format(child_str)
        else:
            return "( )"

    @staticmethod
    def _norm_variable_name(name):
        if "$" in name:
            name = name.replace("$", "s")
        if "?" in name:
            name = name.replace("?", "") + "1"
        return name

    @staticmethod
    def _norm_predicate(name):
        if name == "":
            name = 'rootNode'
        name = name.replace(">", "Greater")
        name = name.replace("<", "Less")
        name = name.replace("<", "Less")
        return re.sub(r'[\.:_]', "-", name)

    @staticmethod
    def _norm_constant(value):
        value = re.sub(r'[\"]', "-", value)
        # value = re.sub(r'[^a-zA-Z\d]', "-", value)
        if re.search(u'[\u4e00-\u9fff]', value):
            value = "chinese-" + re.sub(u'[^a-zA-Z\d]', '-', value)
        value = "\"{}\"".format(value)
        return value

    def to_amr(self, var_exist=None):
        var_exist = var_exist or {}

        if self.is_constant():
            return self._norm_constant(self.value)
        elif self.is_variable_node():
            if var_exist is not None and self.value not in var_exist:
                var_exist[self.value] = self._norm_variable_name(self.value)
                return "({} / var)".format(self._norm_variable_name(self.value))
            else:
                return self._norm_variable_name(self.value)
        else:
            if self.value == "" and len(self.child) == 1:
                amr_str = self.child[0].to_amr(var_exist)
                if amr_str == "\"\"" or amr_str == "":
                    return "(n0 / errorRootNode)"
                elif re.fullmatch(r'".+"', amr_str):
                    return "(n0 / {})".format(amr_str)
                else:
                    return amr_str

            node_name = 'n' + str(len(var_exist))
            amr_str = "({} / {} ".format(node_name, self._norm_predicate(self.value))
            var_exist[node_name] = self._norm_predicate(self.value)
            for i, child in enumerate(self.child):
                child_amr = " :ARG{} ".format(i) + child.to_amr(var_exist)
                amr_str += child_amr
            amr_str += ")"
            return amr_str


def parse_lambda(logic_str: str):
    lg_parent = LogicElement()
    tk_arr = logic_str.split() if isinstance(logic_str, str) else logic_str
    tk_arr = [tk for tk in tk_arr if tk != "," and len(tk) > 0]
    tmp_logic = [lg_parent]
    j = 0
    for i in range(len(tk_arr)):
        if i + j >= len(tk_arr):
            break
        tk = tk_arr[i + j]
        if tk == "(":
            if i + j + 1 < len(tk_arr) and not tk_arr[i + j + 1] == "(":
                new_lg = LogicElement(value=tk_arr[i + j + 1])
                j += 1
            else:
                new_lg = LogicElement()
            tmp_logic[-1].add_child(new_lg)
            tmp_logic.append(new_lg)
        elif tk == ")":
            tmp_logic.pop()
        else:
            tmp_logic[-1].add_child(LogicElement(value=tk))

    return lg_parent if len(lg_parent.child) > 1 else lg_parent.child[0]


def parse_prolog(logic_str: str):
    lg_parent = LogicElement()
    tk_arr = logic_str.split()
    tk_arr = [tk for tk in tk_arr if tk != "," and len(tk) > 0]
    tmp_logic = [lg_parent]
    for i in range(len(tk_arr)):
        tk = tk_arr[i]
        if tk == "(":
            if i > 0 and (tk_arr[i - 1] == "(" or tk_arr[i - 1] == ")"):
                new_lg = LogicElement()
                tmp_logic[-1].add_child(new_lg)
                tmp_logic.append(new_lg)
            pass
        elif tk == ")":
            tmp_logic.pop()
        else:
            if i + 1 < len(tk_arr) and tk_arr[i + 1] == "(":
                new_lg = LogicElement(value=tk_arr[i])
                tmp_logic[-1].add_child(new_lg)
                tmp_logic.append(new_lg)
            else:
                tmp_logic[-1].add_child(LogicElement(value=tk))

    return lg_parent


class StructureTextField(TextMultiField):
    """Text data that have structure such as logical form
    """

    def process(self, batch, device=None):
        """Convert outputs of preprocess into Tensors.

        Args:
            batch (List[List[List[str]]]): A list of length batch size.
                Each element is a list of the preprocess results for each
                field (which are lists of str "words" or feature tags.
            device (torch.device or str): The device on which the tensor(s)
                are built.

        Returns:
            torch.LongTensor or Tuple[LongTensor, LongTensor]:
                A tensor of shape ``(seq_len, batch_size, len(self.fields))``
                where the field features are ordered like ``self.fields``.
                If the base field returns lengths, these are also returned
                and have shape ``(batch_size,)``.
        """

        # batch (list(list(list))): batch_size x len(self.fields) x seq_len
        batch_by_feat = list(zip(*batch))
        base_data = self.base_field.process(batch_by_feat[0], device=device)
        if self.base_field.include_lengths:
            # lengths: batch_size
            base_data, lengths = base_data

        feats = [ff.process(batch_by_feat[i], device=device)
                 for i, (_, ff) in enumerate(self.fields[1:], 1)]
        levels = [base_data] + feats
        # data: seq_len x batch_size x len(self.fields)
        data = torch.stack(levels, 2)
        if self.base_field.include_lengths:
            return data, lengths
        else:
            return data

    def preprocess(self, x):
        """Preprocess data.

        Args:
            x (str): A sentence string (words joined by whitespace).

        Returns:
            List[List[str]]: A list of length ``len(self.fields)`` containing
                lists of tokens/feature tags for the sentence. The output
                is ordered like ``self.fields``.
        """
        return [f.preprocess(x) for _, f in self.fields]


class TreeField(Field):
    def __init__(self, sep=None, **kwargs):
        super(TreeField, self).__init__(**kwargs)
        self.sep = sep

    def preprocess(self, x):
        preprocessed_text = super(TreeField, self).preprocess(x)
        tree_ = parse_lambda(preprocessed_text)
        path_to_leafs = tree_.get_path_to_leaf_nodes()

        tree_linearized = []
        for path_to_leaf in path_to_leafs:
            tree_linearized = tree_linearized + [self.sep] + path_to_leaf
        if len(tree_linearized) > 1:
            tree_linearized = tree_linearized[1:]
        return tree_linearized

    def process(self, batch, device=None):
        """ Process a list of examples to create a torch.Tensor.

        Pad, numericalize, and postprocess a batch and create a tensor.

        Args:
            batch (list(object)): A list of object from a batch of examples.
        Returns:
            torch.autograd.Variable: Processed object given the input
            and custom postprocessing Pipeline.
        """
        batch_size = len(batch)
        sentence_lengths = []
        tree_paths = []
        for s in batch:
            start_idx = 0
            count = 1
            for i, w in enumerate(s):
                if w == self.sep:
                    count += 1
                    tree_paths.append(s[start_idx: i])
                    start_idx = i + 1

            # add final
            tree_paths.append(s[start_idx:])
            sentence_lengths.append(count)

        # padding word level
        max_len = max(sentence_lengths)
        idx_pad = 0
        for i, sent_len in enumerate(sentence_lengths):
            idx_pad = idx_pad + sent_len
            if sent_len < max_len:
                num_pad = max_len - sent_len
                for j in range(num_pad):
                    tree_paths.insert(idx_pad, [self.pad_token])
                idx_pad = idx_pad + num_pad

        padded = self.pad(tree_paths)
        tensor = self.numericalize(padded, device=device)
        # tensor = tensor.reshape(max_len, batch_size, -1)
        return tensor


def structure_text_fields(**kwargs):
    include_lengths = kwargs["include_lengths"]
    base_name = kwargs["base_name"]
    pad = kwargs.get("pad", "<blank>")
    # bos = kwargs.get("bos", "<s>")
    # eos = kwargs.get("eos", "</s>")
    sep = kwargs.get("sep", "<sep>")
    lower = kwargs.get("lower", True)
    truncate = kwargs.get("truncate", None)
    tokenize = partial(
        _feature_tokenize,
        layer=0,
        truncate=truncate,
        feat_delim=None)
    use_len = include_lengths
    base_feat = TreeField(
        sep=sep,
        # init_token=bos, eos_token=eos,
        pad_token=pad, tokenize=tokenize,
        include_lengths=use_len,
        lower=lower
    )

    field = StructureTextField(base_name, base_feat, [])
    return field



