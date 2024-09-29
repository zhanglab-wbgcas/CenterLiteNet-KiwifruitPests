from pathlib import Path

from .easy_set import EasySet

# CUB_SPECS_DIR = Path("/home/liukc/project/dl_plantdisease/meta_learning_pro/fine_grained_pro/github_dir/Few-shot_Imageclassification/data/CUB")

# CUB_SPECS_DIR = Path("/public/home/ZXJ_liukc/few_shot_dir/01multi_weight_pro/data/part_20class_20per_cub")
CUB_SPECS_DIR = Path("/public/home/ZXJ_liukc/few_shot_dir/01multi_weight_pro/data/part_20_cub")


# 确保路径存在
if not CUB_SPECS_DIR.exists():
    raise FileNotFoundError(f"Directory {CUB_SPECS_DIR} does not exist.")

class CUB(EasySet):
    def __init__(self, split: str, **kwargs):
        """
        Build the CUB dataset for the specific split.
        Args:
            split: one of the available split (typically train, val, test).
        Raises:
            ValueError: if the specified split cannot be associated with a JSON spec file
                from CUB's specs directory
        """
        specs_file = CUB_SPECS_DIR / f"{split}.json"
        print(specs_file)
        if not specs_file.is_file():
            raise ValueError(
                f"Could not find specs file {specs_file.name} in {CUB_SPECS_DIR}"
            )
        super().__init__(specs_file=specs_file, **kwargs)
