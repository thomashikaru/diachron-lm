# utility functions for processing historical data
import glob
import re
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters

DATA_PATH = "/Users/thomasclark/mit/diachron-lm/data/coha/dataverse_files"

punkt_param = PunktParameters()
punkt_param.abbrev_types = set(
    [
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "inc",
        "st",
        "vs",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
    ]
)
sentence_splitter = PunktSentenceTokenizer(punkt_param)


def iter_coha_file(filename: str):
    """iterate over COHA documents from a text file

    Args:
        filename (str): path to text file

    Yields:
        str: an individual document
    """
    with open(filename) as f:
        for line in f:
            if len(line.strip()) == 0:
                continue
            yield line.strip()


def iter_coha_decade(decade: int):
    """iterate over COHA documents given a decade

    Args:
        decade (int): name of decade in int form, e.g. 1870 or 1910, must be a multiple of 10 between 1810 and 2000

    Yields:
        str: an individual document
    """
    assert decade in range(1810, 2010, 10), "invalid decade"
    filenames = glob.iglob(f"{DATA_PATH}/{decade}/*.txt")
    for filename in filenames:
        with open(filename) as f:
            header = f.readline()
            assert header.startswith("@")
            space = f.readline()
            assert space == "\n"
            for line in f:
                yield line.strip()


def process_document(document):
    # remove parentheses
    document = re.sub("\s\([^\)]*\)", " ", document)
    document = re.sub("--\s", " ", document)
    document = re.sub("\s+", " ", document)

    sents = sentence_splitter.tokenize(document)
    return sents


if __name__ == "__main__":
    for doc in iter_coha_decade(1810):
        print(*process_document(doc), sep="\n")
