#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

from onmt.utils.logging import init_logger
from onmt.utils.misc import split_corpus
from onmt.translate.translator import build_translator

import onmt.opts as opts
from onmt.utils.parse import ArgumentParser


def translate(opt):
    ArgumentParser.validate_translate_opts(opt)
    logger = init_logger(opt.log_file)

    translator = build_translator(opt, logger=logger, report_score=True)
    src_shards = split_corpus(opt.src, opt.shard_size)
    tgt_shards = split_corpus(opt.tgt, opt.shard_size)

    if hasattr(opt, "soft_templ"):
        soft_templ_shards = split_corpus(opt.soft_templ, opt.shard_size)
        shard_pairs = zip(src_shards, tgt_shards, soft_templ_shards)
        translate_kwargs = {}
        for i, (src_shard, tgt_shard, soft_templ_shard) in enumerate(shard_pairs):
            translate_kwargs["soft_tgt_templ"] = soft_templ_shard
            logger.info("Translating shard %d." % i)
            translator.translate(
                src=src_shard,
                tgt=tgt_shard,
                src_dir=opt.src_dir,
                batch_size=opt.batch_size,
                batch_type=opt.batch_type,
                attn_debug=opt.attn_debug,
                align_debug=opt.align_debug,
                marking_condition=opt.marking_condition,
                self_attn_debug=opt.self_attn_debug,
                self_attn_folder_save="/".join(opt.src.split("/")[:-1]),
                **translate_kwargs
            )
    else:
        shard_pairs = zip(src_shards, tgt_shards)

        for i, (src_shard, tgt_shard) in enumerate(shard_pairs):
            logger.info("Translating shard %d." % i)
            translator.translate(
                src=src_shard,
                tgt=tgt_shard,
                src_dir=opt.src_dir,
                batch_size=opt.batch_size,
                batch_type=opt.batch_type,
                attn_debug=opt.attn_debug,
                align_debug=opt.align_debug,
                marking_condition=opt.marking_condition,
                self_attn_debug=opt.self_attn_debug,
                self_attn_folder_save="/".join(opt.src.split("/")[:-1])
            )


def _get_parser():
    parser = ArgumentParser(description='translate.py')

    opts.config_opts(parser)
    opts.translate_opts(parser)
    return parser


def main():
    parser = _get_parser()

    opt = parser.parse_args()
    translate(opt)


if __name__ == "__main__":
    main()
