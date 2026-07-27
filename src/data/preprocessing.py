import logging
from datasets import load_dataset, concatenate_datasets, Dataset

logger = logging.getLogger(__name__)

def prepare_nli_dataset() -> Dataset:
    logger.info("Loading SNLI and MultiNLI datasets...")
 
    snli = load_dataset("snli", split="train")
    mnli = load_dataset("multi_nli", split="train")

    snli = snli.filter(lambda x: x["label"] != -1)
    mnli = mnli.filter(lambda x: x["label"] != -1)

    snli = snli.rename_columns({"premise": "text_a", "hypothesis": "text_b"})
    mnli = mnli.rename_columns({"premise": "text_a", "hypothesis": "text_b"})

    cols_to_keep = ["text_a", "text_b", "label"]
    snli = snli.select_columns(cols_to_keep)
    mnli = mnli.select_columns(cols_to_keep)

    nli_combined = concatenate_datasets([snli, mnli])
    logger.info(f"NLI dataset ready. Total samples: {len(nli_combined)}")
    
    return nli_combined


def prepare_stsb_dataset(split: str = "train") -> Dataset:
    logger.info(f"Loading STS-B dataset ({split} split)...")
    stsb = load_dataset("glue", "stsb", split=split)
    
    stsb = stsb.rename_columns({"sentence1": "text_a", "sentence2": "text_b"})
    
    def normalize_score(example):
        example["label"] = example["label"] / 5.0
        return example
        
    stsb = stsb.map(normalize_score)
    stsb = stsb.select_columns(["text_a", "text_b", "label"])
    
    logger.info(f"STS-B {split} dataset ready. Total samples: {len(stsb)}")
    return stsb


def prepare_qqp_dataset(split: str = "train") -> Dataset:
    logger.info(f"Loading QQP dataset ({split} split)...")
    qqp = load_dataset("glue", "qqp", split=split)
    
    qqp = qqp.rename_columns({"question1": "text_a", "question2": "text_b"})
    qqp = qqp.select_columns(["text_a", "text_b", "label"])
    
    logger.info(f"QQP {split} dataset ready. Total samples: {len(qqp)}")
    return qqp