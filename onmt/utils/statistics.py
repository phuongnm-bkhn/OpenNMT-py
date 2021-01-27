""" Statistics calculation utility """
from __future__ import division
import time
import math
import sys

from nltk.translate.bleu_score import corpus_bleu

from onmt.utils.logging import logger


class Statistics(object):
    """
    Accumulator for loss statistics.
    Currently calculates:

    * accuracy
    * perplexity
    * elapsed time
    """

    def __init__(self, loss=0, n_words=0, n_correct=0, count_sent=0, count_sent_correct=0, bleu_stats=None):
        self.loss = loss
        self.n_words = n_words
        self.n_correct = n_correct
        self.n_src_words = 0
        self.n_sent = count_sent
        self.n_sent_correct = count_sent_correct
        self.bleu_stats = bleu_stats or {'pred': [], 'tgt': []}
        self.start_time = time.time()

    @staticmethod
    def all_gather_stats(stat, max_size=4096):
        """
        Gather a `Statistics` object accross multiple process/nodes

        Args:
            stat(:obj:Statistics): the statistics object to gather
                accross all processes/nodes
            max_size(int): max buffer size to use

        Returns:
            `Statistics`, the update stats object
        """
        stats = Statistics.all_gather_stats_list([stat], max_size=max_size)
        return stats[0]

    @staticmethod
    def all_gather_stats_list(stat_list, max_size=4096):
        """
        Gather a `Statistics` list accross all processes/nodes

        Args:
            stat_list(list([`Statistics`])): list of statistics objects to
                gather accross all processes/nodes
            max_size(int): max buffer size to use

        Returns:
            our_stats(list([`Statistics`])): list of updated stats
        """
        from torch.distributed import get_rank
        from onmt.utils.distributed import all_gather_list

        # Get a list of world_size lists with len(stat_list) Statistics objects
        all_stats = all_gather_list(stat_list, max_size=max_size)

        our_rank = get_rank()
        our_stats = all_stats[our_rank]
        for other_rank, stats in enumerate(all_stats):
            if other_rank == our_rank:
                continue
            for i, stat in enumerate(stats):
                our_stats[i].update(stat, update_n_src_words=True)
        return our_stats

    def update(self, stat, update_n_src_words=False):
        """
        Update statistics by suming values with another `Statistics` object

        Args:
            stat: another statistic object
            update_n_src_words(bool): whether to update (sum) `n_src_words`
                or not

        """
        self.loss += stat.loss
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct
        self.n_sent += stat.n_sent
        self.n_sent_correct += stat.n_sent_correct
        if self.bleu_stats is not None and stat.bleu_stats is not None:
            self.bleu_stats['tgt'] = self.bleu_stats['tgt'] + stat.bleu_stats['tgt']
            self.bleu_stats['pred'] = self.bleu_stats['pred'] + stat.bleu_stats['pred']

        if update_n_src_words:
            self.n_src_words += stat.n_src_words

    def accuracy(self):
        """ compute accuracy """
        return 100 * (self.n_correct / self.n_words)

    def accuracy_sent(self):
        """ compute accuracy """
        return 100 * (self.n_sent_correct / self.n_sent) if self.n_sent > 0 else 0

    def bleu(self):
        """ compute bleu score """
        if len(self.bleu_stats['tgt']) == len(self.bleu_stats['pred']):
            references = [[x.split()] for x in self.bleu_stats['tgt']]
            candidates = [x.split() for x in self.bleu_stats['pred']]
            return corpus_bleu(references, candidates)

        return 0

    def xent(self):
        """ compute cross entropy """
        return self.loss / self.n_words

    def ppl(self):
        """ compute perplexity """
        return math.exp(min(self.loss / self.n_words, 100))

    def elapsed_time(self):
        """ compute elapsed time """
        return time.time() - self.start_time

    def output(self, step, num_steps, learning_rate, start):
        """Write out statistics to stdout.

        Args:
           step (int): current step
           n_batch (int): total batches
           start (int): start time of step.
        """
        t = self.elapsed_time()
        step_fmt = "%2d" % step
        if num_steps > 0:
            step_fmt = "%s/%5d" % (step_fmt, num_steps)
        logger.info(
            ("Step %s; acc: %6.2f; sent_acc: %6.2f; bleu: %6.4f; ppl: %5.2f; xent: %4.2f; " +
             "lr: %7.5f; %3.0f/%3.0f tok/s; %6.0f sec")
            % (step_fmt,
               self.accuracy(),
               self.accuracy_sent(),
               self.bleu(),
               self.ppl(),
               self.xent(),
               learning_rate,
               self.n_src_words / (t + 1e-5),
               self.n_words / (t + 1e-5),
               time.time() - start))
        sys.stdout.flush()

    def log_tensorboard(self, prefix, writer, learning_rate, step):
        """ display statistics to tensorboard """
        t = self.elapsed_time()
        writer.add_scalar(prefix + "/xent", self.xent(), step)
        writer.add_scalar(prefix + "/ppl", self.ppl(), step)
        writer.add_scalar(prefix + "/accuracy", self.accuracy(), step)
        writer.add_scalar(prefix + "/accuracy_sent", self.accuracy_sent(), step)
        writer.add_scalar(prefix + "/bleu", self.bleu(), step)
        writer.add_scalar(prefix + "/tgtper", self.n_words / t, step)
        writer.add_scalar(prefix + "/lr", learning_rate, step)
