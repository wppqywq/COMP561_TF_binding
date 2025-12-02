TF_NAME = "CTCF"

WINDOW_SIZE = 101

TRAIN_CHROMS = [str(c) for c in range(1, 15)]
VAL_CHROMS = ["15", "16"]
TEST_CHROMS = [str(c) for c in range(17, 23)] + ["X"]

HG19_FASTA_PATH = "/Users/apple/grad_material/2025 fall/comp 561/final/data"
GM12878_BED_PATH = "/Users/apple/grad_material/2025 fall/comp 561/final/data/wgEncodeRegTfbsClusteredV3.GM12878.merged.bed.gz"
FACTORBOOK_POS_PATH = "/Users/apple/grad_material/2025 fall/comp 561/final/data/factorbookMotifPos.txt.gz"


DNA_SHAPE_TRACKS = {
    "MGW": "/absolute/path/to/dnashape/MGW.bw",
    "ProT": "/absolute/path/to/dnashape/ProT.bw",
    "Roll": "/absolute/path/to/dnashape/Roll.bw",
    "HelT": "/absolute/path/to/dnashape/HelT.bw",
}

OUTPUT_DIR = "output"

WINDOW_FASTA_PATH = "/Users/apple/grad_material/2025 fall/comp 561/final/output/ctcf_windows.fa"
WINDOW_META_PATH = "/Users/apple/grad_material/2025 fall/comp 561/final/output/ctcf_windows_meta.tsv"


