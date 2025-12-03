import requests
import zipfile
import os
from deimkit import Trainer, Config, configure_dataset, load_model, list_models, configure_model
from loguru import logger


logger.info("Listing models supported by DEIMKit...")
logger.info(list_models())

logger.info("Configuring model and dataset...")
conf = Config.from_model_name("deim_hgnetv2_n")

conf = configure_dataset(
    config=conf,
    image_size=(640, 640),
    train_ann_file="/kaggle/input/scd1280/train/_annotations.coco.json",
    train_img_folder="/kaggle/input/scd1280/train/",
    val_ann_file="/kaggle/input/scd1280/valid/_annotations.coco.json",
    val_img_folder="/kaggle/input/scd1280/valid/",
    train_batch_size=16,
    val_batch_size=16,
    num_classes=3,
    output_dir="./outputs/coco8/deim_hgnetv2_n",
)

conf = configure_model(conf, pretrained=True, freeze_at=0)

logger.info("Training model...")
trainer = Trainer(conf)

trainer.fit(epochs=30, save_best_only=True)

logger.info("Loading pretrained model...")
model = load_model("deim_hgnetv2_n", image_size=(640, 640))

logger.info("Testing predictions...")
predictions = model.predict("./data/coco8-converted/000000000009.jpg")

logger.info("Loading best model from training...")
model = load_model(
    "deim_hgnetv2_n",
    image_size=(640, 640),
    checkpoint="./outputs/coco8/deim_hgnetv2_n/best.pth",
)

logger.info("Testing predictions...")
predictions = model.predict("/kaggle/input/scd1280/test/frame_1122.jpg")

logger.info("Exporting pretrained model to ONNX...")

coco_classes = ['Background', 'Boat', 'Faust V']

model = load_model("deim_hgnetv2_n", class_names=coco_classes)
model.cfg.save("./checkpoints/deim_hgnetv2_n.yml")

from deimkit.exporter import Exporter
from deimkit.config import Config

config = Config("./checkpoints/deim_hgnetv2_n.yml")
exporter = Exporter(config)

output_path = exporter.to_onnx(
    checkpoint_path="./checkpoints/deim_hgnetv2_n.pth",
    output_path="./checkpoints/deim_hgnetv2_n.onnx"
)

logger.info(f"ONNX model saved to {output_path}")
