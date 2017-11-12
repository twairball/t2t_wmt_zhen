#!/bin/bash

# decoding targets
TMP=tmp/wmt
DECODE_DIR=decode/wmt
DECODE_FILE=$DECODE_DIR/test.en

# model
DATA_DIR=data/wmt
TRAIN_DIR=train/wmt
PROBLEM=translate_enzh_wmt
HPARAMS=transformer_base_single_gpu
MODEL=transformer
TOOLS=./tools

# decode hparams
BEAM_SIZE=4
ALPHA=0.6

mkdir -p $TMP
mkdir -p $DECODE_DIR

# download test dataset if not available
# in this case, we decode on dev dataset

if [ ! -f $DECODE_FILE ]; then
    echo "Decode file not found, downloading..."
    mkdir -p $TMP
    DATASET_URL=http://data.statmt.org/wmt17/translation-task/test.tgz
    wget -N $DATASET_URL -P $TMP/

    echo "Unarchiving..."
    tar xvzf $TMP/test.tgz -C $TMP
    mv $TMP/test/newstest2017-zhen-ref.en.sgm $DECODE_DIR/test.en.sgm
    mv $TMP/test/newstest2017-zhen-src.zh.sgm $DECODE_DIR/test.zh.sgm
    rm -rf $TMP/test

    echo "Preprocessing, removing sgm tags and tokenizing"
    python ./tools/preprocess.py --input $DECODE_DIR/test.en.sgm > $DECODE_FILE
fi


# skip decoding if already cached.
DECODED=$DECODE_FILE.$MODEL.$HPARAMS.$PROBLEM.beam$BEAM_SIZE.alpha$ALPHA.decodes
if [ ! -f $DECODED ]; then 
  echo "Starting Decoder...."
  t2t-decoder \
    --t2t_usr_dir=./data_generators \
    --data_dir=$DATA_DIR \
    --problems=$PROBLEM \
    --model=$MODEL \
    --hparams_set=$HPARAMS \
    --output_dir=$TRAIN_DIR \
    --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA,use_last_position_only=True" \
    --decode_from_file=$DECODE_FILE
  
  echo "Decoded to ${DECODED}"
fi

# postprocess
DECODED_PP=$DECODE_DIR/dev.b$BEAM_SIZE.a$ALPHA.pp.decodes
echo "Post-processing, removing spaces: ${DECODED_PP}"
python $TOOLS/unjieba.py --input $DECODED > $DECODED_PP

# scoring
SRC_FILE=$DECODE_DIR/test.en # without .sgm
REF_FILE=$DECODE_DIR/test.zh
HYP_FILE=$DECODED_PP
SUBM_NAME=SHMLMU_v3.b$BEAM_SIZE.a$ALPHA  # model name that is shown on submission

# wrap hypothesis into sgm file
echo "Wrapping hypothesis into sgm file: ${HYP_FILE}.sgm"
$TOOLS/wrap_xml.pl zh $SRC_FILE.sgm $SUBM_NAME < $HYP_FILE > $HYP_FILE.sgm

# segment reference file and hypothesis file
echo "Segmenting hypothesis file: ${HYP_FILE}.seg.sgm"
$TOOLS/chi_char_segment.pl -t xml < $HYP_FILE.sgm > $HYP_FILE.seg.sgm
echo "Segmenting reference file: ${REF_FILE}.seg.sgm"
$TOOLS/chi_char_segment.pl -t xml < $REF_FILE.sgm > $REF_FILE.seg.sgm

# calculate BLEU score
echo "Calculating BLEU score: ${HYP_FILE}.bleu.log"
$TOOLS/mteval-v11b.pl -s $SRC_FILE.sgm -r $REF_FILE.seg.sgm -t $HYP_FILE.seg.sgm -c > $HYP_FILE.bleu.log


