DECADE_PREFIXES = ["181", "182", "183", "184", "185", "186", "187", "188", "189", "190", 
    "191", "192", "193", "194", "195", "196", "197", "198", "199", "200"]
DECADES = list(range(1810, 2010, 10))

rule download_coha:
    output:
        "data/coha/dataverse_files.zip"
    resources:
        mem_mb=1000,
        runtime=720
    shell:
        """
        mkdir -p /om2/user/thclark/coha
        cd /om2/user/thclark/coha
        wget https://rserve.dataverse.harvard.edu/cgi-bin/zipdownload?b12-f0308d1f48b2
        """

rule unzip_coha:
    input:
        "data/coha/dataverse_files.zip"
    output:
        "data/coha/dataverse_files/fic_1810_8641.txt",
        "data/coha/dataverse_files/fic_2000_27727.txt"
    resources:
        mem_mb=1000,
        runtime=720
    shell:
        """
        cd data/coha
        mkdir -p dataverse_files
        unzip dataverse_files.zip -d dataverse_files
        cd dataverse_files
        unzip 'text_*.zip'
        """

rule make_dirs:
    input:
        "data/coha/dataverse_files/fic_1810_8641.txt",
        "data/coha/dataverse_files/fic_2000_27727.txt"
    output:
        "data/coha/dataverse_files/1810/fic_1810_8641.txt",
        "data/coha/dataverse_files/2000/fic_2000_27727.txt",
    shell:
        """
        cd data/coha/dataverse_files
        mkdir -p 1810
        mv *_181*_*.txt 1810
        mkdir -p 1820
        mv *_182*_*.txt 1820
        mkdir -p 1830
        mv *_183*_*.txt 1830
        mkdir -p 1840
        mv *_184*_*.txt 1840
        mkdir -p 1850
        mv *_185*_*.txt 1850
        mkdir -p 1860
        mv *_186*_*.txt 1860
        mkdir -p 1870
        mv *_187*_*.txt 1870
        mkdir -p 1880
        mv *_188*_*.txt 1880
        mkdir -p 1890
        mv *_189*_*.txt 1890
        mkdir -p 1900
        mv *_190*_*.txt 1900
        mkdir -p 1910
        mv *_191*_*.txt 1910
        mkdir -p 1920
        mv *_192*_*.txt 1920
        mkdir -p 1930
        mv *_193*_*.txt 1930
        mkdir -p 1940
        mv *_194*_*.txt 1940
        mkdir -p 1950
        mv *_195*_*.txt 1950
        mkdir -p 1960
        mv *_196*_*.txt 1960
        mkdir -p 1970
        mv *_197*_*.txt 1970
        mkdir -p 1980
        mv *_198*_*.txt 1980
        mkdir -p 1990
        mv *_199*_*.txt 1990
        mkdir -p 2000
        mv *_200*_*.txt 2000
        """

rule get_coha_stats:
    input:
    output:
    run:
        import glob

        f = open("data/coha/stats.txt")

        for decade_prefix in DECADE_PREFIXES:
            filenames = glob.glob(f"*_{decade_prefix}*_*.txt")

        f.close()

rule clean_and_sample:
    input:
        "data/coha/dataverse_files/{decade}.txt"
    output:
        "data/coha/lm_data/{decade}/en.train",
        "data/coha/lm_data/{decade}/en.test",
        "data/coha/lm_data/{decade}/en.valid",
    resources:
        mem_mb=4000,
        runtime=720
    shell:
        """
        cd src
        python sample_coha.py --decade {wildcards.decade} --output_dir ../data/coha/lm_data/{wildcards.decade}
        """


# train bpe on each decade's data
rule train_transformer_bpe:
    input:
        "data/coha/lm_data/{decade}/en.train"
    output:
        "models/bpe_codes/30k/{decade}/en.codes"
    resources:
        mem_mb=16000,
        runtime=60
    conda:
        "rus"
    log:
        "logs/train_bpe_{decade}.log"
    shell:
        """
        OUTPATH="models/bpe_codes/30k/{wildcards.decade}"  # path where processed files will be stored
        FASTBPE=fastBPE/fast  # path to the fastBPE tool
        NUM_OPS=30000
        INPUT_DIR="data/coha/{wildcards.decade}"

        # create output path
        mkdir -p $OUTPATH

        lang="en"
        # learn bpe codes on the training set (or only use a subset of it)
        $FASTBPE learnbpe $NUM_OPS $INPUT_DIR/$lang.train > $OUTPATH/$lang.codes
        """

# apply bpe to each decade's data
rule apply_transformer_bpe:
    input:
        "data/coha/lm_data/{decade}/en.train",
        "data/coha/lm_data/{decade}/en.valid",
        "data/coha/lm_data/{decade}/en.test",
        "models/bpe_codes/30k/{decade}/en.codes"
    output:
        "data/coha/lm_data/{decade}/en-bpe/en.train",
        "data/coha/lm_data/{decade}/en-bpe/en.valid",
        "data/coha/lm_data/{decade}/en-bpe/en.test",
    resources:
        mem_mb=16000,
        runtime=60
    conda:
        "rus"
    log:
        "logs/apply_bpe_{decade}.log"
    shell:
        """
        BPE_CODES="models/bpe_codes/30k/{wildcards.decade}"  # path where processed files will be stored
        FASTBPE="fastBPE/fast"  # path to the fastBPE tool
        INPUT_DIR="data/coha/lm_data/{wildcards.decade}"
        OUT_DIR="data/coha/lm_data/{wildcards.decade}/en-bpe"

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
        "data/coha/lm_data/{decade}/en-bpe/en.train",
        "data/coha/lm_data/{decade}/en-bpe/en.valid",
        "data/coha/lm_data/{decade}/en-bpe/en.test",
    output:
        "data/coha/lm_data/{decade}/en-bin/train.bin",
        "data/coha/lm_data/{decade}/en-bin/valid.bin",
        "data/coha/lm_data/{decade}/en-bin/test.bin",
    resources:
        mem_mb=16000,
        runtime=120
    conda:
        "rus"
    log:
        "logs/preprocess_data_{decade}.log"
    shell:
        """
        data_dir="data/coha/lm_data/{wildcards.decade}/en-bpe"
        out_dir="data/coha/lm_data/{wildcards.decade}/en-bin"
        fairseq-preprocess \
                --only-source \
                --trainpref $data_dir/en.train \
                --validpref $data_dir/en.valid \
                --testpref $data_dir/en.test \
                --destdir $out_dir \
                --bpe fastbpe \
                --workers 20
        """

# train a Transformer language model on each decade's data
rule train_transformer_lm:
    input:
        "data/coha/lm_data/{decade}/en-bin/train.bin",
        "data/coha/lm_data/{decade}/en-bin/valid.bin",
        "data/coha/lm_data/{decade}/en-bin/test.bin",
    output:
        "models/{decade}/checkpoint_last.pt"
    resources:
        mem_mb=20000,
        runtime=2160,
        slurm_extra="--gres=gpu:1"
    conda:
        "rus"
    log:
        "logs/train_transformer_{decade}.log"
    shell:
        """
        DATA_DIR="data/coha/lm_data/{wildcards.decade}/en-bin"
        SAVE_DIR="models/{wildcards.decade}"
        default=1
        RANDOM_SEED="${{3:-$default}}"

        mkdir -p models/{wildcards.decade}

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