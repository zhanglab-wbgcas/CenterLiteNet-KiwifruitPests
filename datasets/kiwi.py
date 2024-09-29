

from pathlib import Path

from .easy_set import EasySet


kiwi_SPECS_DIR = Path("/public/home/ZXJ_liukc/few_shot_dir/01multi_weight_pro/data/kiwi_pest_data")

# 确保路径存在
if not kiwi_SPECS_DIR.exists():
    raise FileNotFoundError(f"Directory {kiwi_SPECS_DIR} does not exist.")

class KIWI(EasySet):
    def __init__(self, split: str, **kwargs):
        """
        Build the kiwi dataset for the specific split.
        Args:
            split: one of the available split (typically train, val, test).
        Raises:
            ValueError: if the specified split cannot be associated with a JSON spec file
                from kiwi's specs directory
        """
        specs_file = kiwi_SPECS_DIR / f"{split}.json"
        print(specs_file)
        if not specs_file.is_file():
            raise ValueError(
                f"Could not find specs file {specs_file.name} in {kiwi_SPECS_DIR}"
            )
        super().__init__(specs_file=specs_file, **kwargs)