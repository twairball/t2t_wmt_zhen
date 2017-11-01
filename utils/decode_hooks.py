#!/usr/bin/env python
# -*- coding: utf-8 -*-

from six.moves import zip

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util

class DecodeHook(session_run_hook.SessionRunHook):    
    """Prints decoded sentences every N local steps, or at end. 
    params:
        source: array of source sentences
        target: array of target sentences
        output_dir: 
        output_filename: 
        every_n_iter: decode sentences every N interations
    """

    def __init__(self, source, target, 
        output_dir, output_filename="decode.out", 
        every_n_iter=2500):
        self.source = source
        self.target = target
        self.output_dir = output_dir
        self.output_filename = output_filename
        self.every_n_iter = every_n_iter
    
    def begin(self):
        self._iter_count = 0
    
    def after_run(self, run_context, run_values):
        sess = run_context.session
        step = self._iter_count
        if self._should_trigger_for_step(step):
            decoded = self._decode(sess, step)
            self.print_decoded(decoded, step)
        
        self._iter_count += 1

    def print_decoded(self, decoded, step):
        formatted = ["src: %s\nref: %s\nhyp: %s\n" % (src, ref, hyp) 
            for src, ref, hyp in zip(self.source, self.target, decoded)]
        report = "[decode] step=%d \n" % step
        report += "\n".join(formatted)
        logging.info(report)

    def _should_trigger_for_step(self, step):
        return (step % self.every_n_iter) == 0

    def _decode(self, session, step):
        return []
    
    def _save(self, session, step):
        """Save decoded output"""
        save_path = "%s/%s.%d" % (self.output_dir, self.output_filename, step)
        logging.info("Saving checkpoints for %d into %s.", step, save_path)

        # TODO
