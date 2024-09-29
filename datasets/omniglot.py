from pathlib import Path
import sys
# sys.path.append('/home/liukc/project/dl_plantdisease/meta_learning_pro/fine_grained_pro/github_dir/easy-few-shot-learning/easyfsl/datasets')

from .easy_set import EasySet

Omniglot_SPECS_DIR = Path("/home/liukc/project/dl_plantdisease/meta_learning_pro/fine_grained_pro/github_dir/easy-few-shot-learning/data/Omniglot")


class Omniglot(EasySet):
    def __init__(self, split: str, **kwargs):
        """
        Build the Omniglot dataset for the specific split.
        Args:
            split: one of the available split (typically train, val, test).
        Raises:
            ValueError: if the specified split cannot be associated with a JSON spec file
                from Omniglot's specs directory
        """
        specs_file = Omniglot_SPECS_DIR / f"{split}.json"
        print(specs_file)
        if not specs_file.is_file():
            raise ValueError(
                f"Could not find specs file {specs_file.name} in {Omniglot_SPECS_DIR}"
            )
        super().__init__(specs_file=specs_file, **kwargs)
