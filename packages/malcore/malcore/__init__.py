"""malcore: constants, hashing, provenance and metrics shared by malprep, malexp and malanalysis.

Stage indices are fixed here and used by every package: ring = 0, trophozoite = 1,
schizont = 2, gametocyte = 3. The rare-stage group L = {2, 3} (protocol Step 6).
"""
__version__ = "2.0.0"

STAGES = ["R", "T", "S", "G"]
STAGE_NAMES = {"R": "ring", "T": "trophozoite", "S": "schizont", "G": "gametocyte"}
SPECIES = ["falciparum", "vivax", "ovale", "malariae"]
EARLY = (0, 1)          # ring, trophozoite
RARE = (2, 3)           # schizont, gametocyte = the rare-stage group L
N_STAGES = 4

ARMS_BASE = ["B0", "B1", "B2", "B3", "B4"]
ARMS_DERIVED = ["B5", "B6", "M-aux", "M-full", "M-prob"]
ARMS_ABLATION = ["ABL-noaux", "ABL-detach", "ABL-coordattn"]
ARMS = ARMS_BASE + ARMS_DERIVED + ARMS_ABLATION
