rule download_coha:
    output:
    resources:
        mem_mb=4000,
        runtime=720
    shell:
        """
        mkdir -p /om2/user/thclark/coha
        cd /om2/user/thclark/coha
        wget https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/8SRSYK#
        """

# train bpe on each decade's data
rule train_transformer_bpe:
    input:
        "data/coha/{decade}/ru.train"
    output:
        "models/bpe_codes/30k/ru.codes"
    resources:
        mem_mb=16000,
        runtime=60
    conda:
        "rus"
    log:
        "logs/train_bpe_{decade}.log"
    shell:
        """
        OUTPATH="models/bpe_codes/30k"  # path where processed files will be stored
        FASTBPE=fastBPE/fast  # path to the fastBPE tool
        NUM_OPS=30000
        INPUT_DIR="data/coha/{decade}"

        # create output path
        mkdir -p $OUTPATH

        lang="en"
        # learn bpe codes on the training set (or only use a subset of it)
        $FASTBPE learnbpe $NUM_OPS $INPUT_DIR/$lang.train > $OUTPATH/$lang.codes
        """

# apply bpe to each decade's data
rule apply_transformer_bpe:
    input:
        "data/coha/{decade}/en.train",
        "data/coha/{decade}/en.valid",
        "data/coha/{decade}/en.test",
        "models/bpe_codes/30k/en.codes"
    output:
        "data/coha/{decade}/en-bpe/en.train",
        "data/coha/{decade}/en-bpe/en.valid",
        "data/coha/{decade}/en-bpe/en.test",
    resources:
        mem_mb=16000,
        runtime=60
    conda:
        "rus"
    log:
        "logs/apply_bpe_{decade}.log"
    shell:
        """
        BPE_CODES="models/bpe_codes/30k"  # path where processed files will be stored
        FASTBPE="fastBPE/fast"  # path to the fastBPE tool
        INPUT_DIR="ru_data/russian_corpora/counterfactual/transformer"
        OUT_DIR="ru_data/russian_corpora/counterfactual/transformer/ru-bpe"

        # create output path
        mkdir -p $OUT_DIR

        lang="ru"
        extlist=("train" "test" "valid")

        for ext in "${{extlist[@]}}"
        do
            $FASTBPE applybpe $OUT_DIR/$lang.$ext $INPUT_DIR/$lang.$ext $BPE_CODES/$lang.codes
        done
        """

# preprocess/binarize each decade's data
rule preprocess_data_transformer:
    input:
        "ru_data/russian_corpora/counterfactual/transformer/ru-bpe/ru.train",
        "ru_data/russian_corpora/counterfactual/transformer/ru-bpe/ru.test",
        "ru_data/russian_corpora/counterfactual/transformer/ru-bpe/ru.valid",
    output:
        "ru_data/russian_corpora/counterfactual/transformer/ru-bin/train.bin",
        "ru_data/russian_corpora/counterfactual/transformer/ru-bin/test.bin",
        "ru_data/russian_corpora/counterfactual/transformer/ru-bin/valid.bin",
    resources:
        mem_mb=16000,
        runtime=120
    conda:
        "rus"
    log:
        "logs/preprocess_data.log"
    shell:
        """
        data_dir="ru_data/russian_corpora/counterfactual/transformer/ru-bpe"
        out_dir="ru_data/russian_corpora/counterfactual/transformer/ru-bin"
        fairseq-preprocess \
                --only-source \
                --trainpref $data_dir/ru.train \
                --validpref $data_dir/ru.valid \
                --testpref $data_dir/ru.test \
                --destdir $out_dir \
                --bpe fastbpe \
                --workers 20
        """

# train a Transformer language model on each decade's data
rule train_transformer_lm:
    input:
        "ru_data/russian_corpora/counterfactual/transformer/ru-bin/train.bin",
        "ru_data/russian_corpora/counterfactual/transformer/ru-bin/test.bin",
        "ru_data/russian_corpora/counterfactual/transformer/ru-bin/valid.bin",
    output:
        "models/transformer/checkpoint_last.pt"
    resources:
        mem_mb=20000,
        runtime=2160,
        slurm_extra="--gres=gpu:1"
    conda:
        "rus"
    log:
        "logs/train_transformer.log"
    shell:
        """
        DATA_DIR="ru_data/russian_corpora/counterfactual/transformer/ru-bin"
        SAVE_DIR="models/transformer"
        default=1
        RANDOM_SEED="${{3:-$default}}"

        mkdir -p models/transformer

        fairseq-train --task language_modeling \
            $DATA_DIR \
            --save-dir $SAVE_DIR \
            --arch transformer_lm --share-decoder-input-output-embed \
            --dropout 0.1 \
            --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
            --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
            --tokens-per-sample 128 --sample-break-mode none \
            --max-tokens 128 --update-freq 64 \
            --fp16 \
            --max-update 50000 --max-epoch 35 --patience 3  \
            --seed $RANDOM_SEED	  
        """