from trident.segmentation_models import segmentation_model_factory
from trident.patch_encoder_models.load import (
    encoder_factory as patch_encoder_model_factory,
)
from trident.slide_encoder_models.load import (
    encoder_factory as slide_encoder_model_factory,
)

# Download weights for all segmentation models
segmentation_models = ["hest", "grandqc", "grandqc_artifact"]

for model in segmentation_models:
    try:
        segmentation_model_factory(model)
    except Exception as e:
        print(f"Failed to download weights for {model}: {e}")
        continue


# Download weights for all patch encoder models
patch_encoder_models = [
    "conch_v1",
    "uni_v1",
    "uni_v2",
    "ctranspath",
    "phikon",
    "resnet50",
    "gigapath",
    "virchow",
    "virchow2",
    "hoptimus0",
    "hoptimus1",
    "phikon_v2",
    "conch_v15",
    "musk",
    "hibou_l",
    "kaiko-vits8",
    "kaiko-vits16",
    "kaiko-vitb8",
    "kaiko-vitb16",
    "kaiko-vitl14",
    "lunit-vits8",
]

for model in patch_encoder_models:
    try:
        patch_encoder_model_factory(model)
    except Exception as e:
        print(f"Failed to download weights for {model}: {e}")
        continue


# Download weights for all slide encoder models
slide_encoder_models = [
    "threads",
    "titan",
    "prism",
    "gigapath",
    "chief",
    "madeleine",
    "mean-virchow",
    "mean-virchow2",
    "mean-conch_v1",
    "mean-conch_v15",
    "mean-ctranspath",
    "mean-gigapath",
    "mean-resnet50",
    "mean-hoptimus0",
    "mean-phikon",
    "mean-phikon_v2",
    "mean-musk",
    "mean-uni_v1",
    "mean-uni_v2",
]

for model in slide_encoder_models:
    try:
        slide_encoder_model_factory(model)
    except Exception as e:
        print(f"Failed to download weights for {model}: {e}")
        continue


# XDG_CACHE_HOME="<YOUR_CACHE_DIR>" HF_TOKEN="<YOUR_HUGGINGFACE_TOKEN>" python3 run_predownload_weights.py