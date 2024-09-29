# from pathlib import Path

# from .easy_set import EasySet


# # CIFAR_SPECS_DIR = Path("/home/liukc/project/dl_plantdisease/meta_learning_pro/fine_grained_pro/github_dir/Few-shot_Imageclassification/data/part_cifar")
# CIFAR_SPECS_DIR = Path("/home/liukc/project/dl_plantdisease/meta_learning_pro/fine_grained_pro/github_dir/Few-shot_Imageclassification/data/part_20_cifar")

# # 确保路径存在
# if not CIFAR_SPECS_DIR.exists():
#     raise FileNotFoundError(f"Directory {CIFAR_SPECS_DIR} does not exist.")

# class CIFAR(EasySet):
#     def __init__(self, split: str, **kwargs):
#         """
#         Build the CIFAR dataset for the specific split.
#         Args:
#             split: one of the available split (typically train, val, test).
#         Raises:
#             ValueError: if the specified split cannot be associated with a JSON spec file
#                 from CIFAR's specs directory
#         """
#         specs_file = CIFAR_SPECS_DIR / f"{split}.json"
#         print(specs_file)
#         if not specs_file.is_file():
#             raise ValueError(
#                 f"Could not find specs file {specs_file.name} in {CIFAR_SPECS_DIR}"
#             )
#         super().__init__(specs_file=specs_file, **kwargs)


from pathlib import Path

from .easy_set import EasySet


# cifar_SPECS_DIR = Path("/public/home/ZXJ_liukc/few_shot_dir/01multi_weight_pro/data/part_25class_20per_cifar")
cifar_SPECS_DIR = Path("/public/home/ZXJ_liukc/few_shot_dir/01multi_weight_pro/data/part_20_cifar")

# 确保路径存在
if not cifar_SPECS_DIR.exists():
    raise FileNotFoundError(f"Directory {cifar_SPECS_DIR} does not exist.")

class CIFAR(EasySet):
    def __init__(self, split: str, **kwargs):
        """
        Build the cifar dataset for the specific split.
        Args:
            split: one of the available split (typically train, val, test).
        Raises:
            ValueError: if the specified split cannot be associated with a JSON spec file
                from cifar's specs directory
        """
        specs_file = cifar_SPECS_DIR / f"{split}.json"
        print(specs_file)
        if not specs_file.is_file():
            raise ValueError(
                f"Could not find specs file {specs_file.name} in {cifar_SPECS_DIR}"
            )
        super().__init__(specs_file=specs_file, **kwargs)