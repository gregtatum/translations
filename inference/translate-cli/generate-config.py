import yaml
from pathlib import Path


def generate_config():
    config = {
        "relative-paths": True,
        "models": ["model.intgemm.alphas.bin"],
        "vocabs": ["vocab.deen.spm", "vocab.deen.spm"],
        "beam-size": "1",
        "normalize": "1.0",
        "word-penalty": "0",
        "max-length-break": "128",
        "mini-batch-words": "1024",
        "workspace": "128",
        "max-length-factor": "2.0",
        "skip-cost": True,
        "cpu-threads": "0",
        "quiet": "true",
        "quiet-translation": "true",
        "gemm-precision": "int8shiftAlphaAll",
        "alignment": "soft",
    }
    with Path("cli-config.yml").open("w") as outfile:
        yaml.safe_dump(config, outfile)
