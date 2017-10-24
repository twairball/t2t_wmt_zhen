
import os
import random
from . import utils
import tensorflow as tf

def preprocess_nc(data_dir, dataset_prefix="train.full.tok", 
    train_prefix="train.tok", dev_prefix="dev.tok", dev_size=2000):
    # full
    src = os.path.join(data_dir, "%s.en" % dataset_prefix)
    ref = os.path.join(data_dir, "%s.zh" % dataset_prefix)
    # train
    src_train = os.path.join(data_dir, "%s.en" % train_prefix)
    ref_train = os.path.join(data_dir, "%s.zh" % train_prefix)
    # dev
    src_dev = os.path.join(data_dir, "%s.en" % dev_prefix)
    ref_dev = os.path.join(data_dir, "%s.zh" % dev_prefix)

    if utils.do_files_exist([src_train, ref_train, src_dev, ref_dev]):
        return

    # merge blanks
    merge_blanks_and_write(src, ref, src, ref)
    
    # split dataset --> train, dev
    split_dev([src, ref], [src_train, ref_train], [src_dev, ref_dev], sample_size=dev_size)


def split_dev(corpus_filepaths, 
    train_filepaths=["train.tok.en", "train.tok.zh"], 
    dev_filepaths=["dev.tok.en", "dev.tok.zh"],
    sample_size=2000):

    src_filename, ref_filename = corpus_filepaths
    src_train, ref_train = train_filepaths
    src_dev, ref_dev = dev_filepaths
    
    with open(src_filename, 'r') as f_src, \
        open(ref_filename, 'r') as f_ref, \
        open(src_train, 'w') as f_train_src, \
        open(ref_train, 'w') as f_train_ref, \
        open(src_dev, 'w') as f_dev_src, \
        open(ref_dev, 'w') as f_dev_ref:

        lines_src = f_src.readlines()
        lines_ref = f_ref.readlines()

        # shuffle
        index_shuf = random.sample(list(range(len(lines_src))), len(lines_src))
        N = min(len(lines_src), sample_size)

        tf.logging.info("Sample size: %d, total corpus: %d" %
                        (N, len(lines_src)))
        tf.logging.info("Writing dev src to: %s" % src_dev)
        tf.logging.info("Writing dev ref to: %s" % ref_dev)

        for i in index_shuf[:N]:
            f_dev_src.write(lines_src[i])
            f_dev_ref.write(lines_ref[i])

        # write remaining to train
        tf.logging.info("Writing train src to: %s" % src_train)
        tf.logging.info("Writing train ref to: %s" % ref_train)

        for i in index_shuf[N:]:
            f_train_src.write(lines_src[i])
            f_train_ref.write(lines_ref[i])


def merge_blanks_and_write(src, ref, output_src, output_ref):
    src_lines, ref_lines = _merge_blanks(src, ref, verbose=True)

    tf.logging.info("writing to: %s" % output_src)
    with io.open(output_src, 'w', encoding='utf8') as f:
        for l in src_lines:
            f.write(l + os.linesep)

    tf.logging.info("writing to: %s" % output_ref)
    with io.open(output_ref, 'w', encoding='utf8') as f:
        for l in ref_lines:
            f.write(l + os.linesep)
    
def _merge_blanks(src, targ, verbose=False):
    """Read parallel corpus 2 lines at a time. 
    Merge both sentences if only either source or target has blank 2nd line. 
    If both have blank 2nd lines, then ignore. 
    
    Returns tuple (src_lines, targ_lines), arrays of strings sentences. 
    """
    merges_done = [] # array of indices of rows merged
    sub = None # replace sentence after merge
    
    with open(src, 'rb') as src_file, open(targ, 'rb') as targ_file: 
        src_lines = src_file.readlines()
        targ_lines = targ_file.readlines()
        
        print("src: %d, targ: %d" % (len(src_lines), len(targ_lines)))
        print("=" * 30)
        for i in range(0, len(src_lines) - 1):
            s = src_lines[i].decode('utf-8').rstrip()
            s_next = src_lines[i+1].decode('utf-8').rstrip()
            
            t = targ_lines[i].decode('utf-8').rstrip()
            t_next = targ_lines[i+1].decode('utf-8').rstrip()
            
            
            if t == '.':
                t = '' 
            if t_next == '.':
                t_next = ''
                
            if (len(s_next) == 0) and (len(t_next) > 0):
                targ_lines[i] = "%s %s" % (t, t_next) # assume it has punctuation
                targ_lines[i+1] = b''
                src_lines[i] = s if len(s) > 0 else sub
                
                merges_done.append(i)
                if verbose: 
                    print("t [%d] src: %s\n      targ: %s" % (i, src_lines[i], targ_lines[i]))
                    print()
                
            elif (len(s_next) > 0) and (len(t_next) == 0):
                src_lines[i] = "%s %s" % (s, s_next) # assume it has punctuation
                src_lines[i+1] = b''
                targ_lines[i] = t if len(t) > 0 else sub
                
                merges_done.append(i)
                if verbose:
                    print("s [%d] src: %s\n      targ: %s" % (i, src_lines[i], targ_lines[i]))
                    print()
            elif (len(s) == 0) and (len(t) == 0):
                # both blank -- remove
                merges_done.append(i)
            else:
                src_lines[i] = s if len(s) > 0 else sub
                targ_lines[i] = t if len(t) > 0 else sub
                
        # handle last line
        s_last = src_lines[-1].decode('utf-8').strip()
        t_last = targ_lines[-1].decode('utf-8').strip()
        if (len(s_last) == 0) and (len(t_last) == 0):
            merges_done.append(len(src_lines) - 1)
        else:
            src_lines[-1] = s_last
            targ_lines[-1] = t_last
            
    # remove empty sentences
    for m in reversed(merges_done):
        del src_lines[m]
        del targ_lines[m]
    
    print("merges done: %d" % len(merges_done))
    return (src_lines, targ_lines)