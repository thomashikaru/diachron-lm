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
        "data/coha/dataverse_files/unzip_coha.done"
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
        touch unzip_coha.done
        """

rule make_dirs:
    input:
        "data/coha/dataverse_files/unzip_coha.done"
    output:
        "data/coha/dataverse_files/1810/fic_1810_8641.txt",
        "data/coha/dataverse_files/2000/fic_2000_27727.txt",
    shell:
        """
        cd data/coha/dataverse_files
        mkdir -p 1810
        files=( *_181*_*.txt); [ -f $files ] && echo ${{files[*]}} | xargs mv -t 1810 --
        mkdir -p 1820
        files=( *_182*_*.txt); [ -f $files ] && echo ${{files[*]}} | xargs mv -t 1820 --
        mkdir -p 1830
        files=( *_183*_*.txt); [ -f $files ] && echo ${{files[*]}} | xargs mv -t 1830 --
        mkdir -p 1840
        files=( *_184*_*.txt); [ -f $files ] && echo ${{files[*]}} | xargs mv -t 1840 --
        mkdir -p 1850
        files=( *_185*_*.txt); [ -f $files ] && echo ${{files[*]}} | xargs mv -t 1850 --
        mkdir -p 1860
        files=( *_186*_*.txt); [ -f $files ] && echo ${{files[*]}} | xargs mv -t 1860 --
        mkdir -p 1870
        files=( *_187*_*.txt); [ -f $files ] && echo ${{files[*]}} | xargs mv -t 1870 --
        mkdir -p 1880
        files=( *_188*_*.txt); [ -f $files ] && echo ${{files[*]}} | xargs mv -t 1880 --
        mkdir -p 1890
        files=( *_189*_*.txt); [ -f $files ] && echo ${{files[*]}} | xargs mv -t 1890 --
        mkdir -p 1900
        files=( *_190*_*.txt); [ -f $files ] && echo ${{files[*]}} | xargs mv -t 1900 --
        mkdir -p 1910
        files=( *_191*_*.txt); [ -f $files ] && echo ${{files[*]}} | xargs mv -t 1910 --
        mkdir -p 1920
        files=( *_192*_*.txt); [ -f $files ] && echo ${{files[*]}} | xargs mv -t 1920 --
        mkdir -p 1930
        files=( *_193*_*.txt); [ -f $files ] && echo ${{files[*]}} | xargs mv -t 1930 --
        mkdir -p 1940
        files=( *_194*_*.txt); [ -f $files ] && echo ${{files[*]}} | xargs mv -t 1940 --
        mkdir -p 1950
        files=( *_195*_*.txt); [ -f $files ] && echo ${{files[*]}} | xargs mv -t 1950 --
        mkdir -p 1960
        files=( *_196*_*.txt); [ -f $files ] && echo ${{files[*]}} | xargs mv -t 1960 --
        mkdir -p 1970
        files=( *_197*_*.txt); [ -f $files ] && echo ${{files[*]}} | xargs mv -t 1970 --
        mkdir -p 1980
        files=( *_198*_*.txt); [ -f $files ] && echo ${{files[*]}} | xargs mv -t 1980 --
        mkdir -p 1990
        files=( *_199*_*.txt); [ -f $files ] && echo ${{files[*]}} | xargs mv -t 1990 --
        mkdir -p 2000
        files=( *_200*_*.txt); [ -f $files ] && echo ${{files[*]}} | xargs mv -t 2000 --
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
        python sample.py --decade {wildcards.decade} --output_dir ../data/coha/lm_data/{wildcards.decade} --data_dir ../data/coha/dataverse_files
        """

rule clean_and_sample_all:
    input:
        expand("data/coha/lm_data/{decade}/en.{suff}", decade=DECADES, suff=["train", "test", "valid"])

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