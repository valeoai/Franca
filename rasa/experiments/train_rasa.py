import os
from datetime import datetime

import click
import pandas as pd
import sacred
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger
from torchvision.transforms import (
    ColorJitter,
    Compose,
    Normalize,
    RandomApply,
    RandomGrayscale,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomRotation,
    RandomVerticalFlip,
    Resize,
    ToTensor,
)
from torchvision.transforms.functional import InterpolationMode

from rasa.data.coco.coco_data_module import CocoDataModule
from rasa.data.imagenet.imagenet_data_module_webdataset import ImageNetDataModule
from rasa.data.VOCdevkit.vocdata import TrainXVOCValDataModule, VOCDataModule
from rasa.src.rasa import RASA
from rasa.src.transforms import GaussianBlur

ex = sacred.experiment.Experiment()
api_key = "<enter your api key>"


@click.command()
@click.option("--config_path", type=str)
@click.option("--seed", type=int, default=400)
@click.option("--lr_heads", type=float, default=None)
@click.option("--checkpoint_dir", type=str, default=None)
@click.option("--data_dir", type=str, default=None)
def entry_script(config_path, seed, lr_heads, checkpoint_dir, data_dir):
    if config_path is not None:
        ex.add_config(os.path.join(os.path.abspath(os.path.dirname(__file__)), config_path))
    else:
        ex.add_config(os.path.join(os.path.abspath(os.path.dirname(__file__)), "rasa_config_dev.yml"))
    time = datetime.now().strftime("%Y%m%d-%H%M%S")
    ex_name = f"posentangle-{time}"
    checkpoint_dir = os.path.join(ex.configurations[0]._conf["train"]["checkpoint_dir"], ex_name)
    ex.observers.append(sacred.observers.FileStorageObserver(checkpoint_dir))
    config_updates = {"seed": seed}
    if lr_heads is not None:
        config_updates["train.lr_heads"] = lr_heads
    if checkpoint_dir is not None:
        config_updates["train.checkpoint_dir"] = checkpoint_dir
    if data_dir is not None:
        config_updates["data.data_dir"] = data_dir

    ex.run(config_updates=config_updates, options={"--name": ex_name})


@ex.main
@ex.capture
def finetune_with_spatial_loss(_config, _run):  # finetune_pos_disentanglement
    # Setup logging
    print("Online mode")
    neptune_logger = NeptuneLogger(
        api_key=api_key,
        mode="offline" if _config.get("log_status") == "offline" else "async",
        project="valentinospariza/Posentangle",
        name=_run.experiment_info["name"],
        tags=_config["tags"].split(","),
    )
    # Add parameters to Neptune logger
    params = pd.json_normalize(_config).to_dict(orient="records")[0]
    neptune_logger.experiment["parameters"] = params

    # Process config
    print("Config:")
    print(_config)
    data_config = _config["data"]
    train_config = _config["train"]
    val_config = _config["val"]
    seed_everything(_config["seed"])

    py_light_chkpt = train_config["continue"]
    prev_model = None
    for i in range(train_config["start_pos_layers"], train_config["end_pos_layers"]):
        data_module, num_images = get_training_data(data_config, train_config, val_config, _config["num_workers"])

        # Init method
        model = RASA(
            weight_decay_end=train_config["weight_decay_end"],
            num_clusters_kmeans=val_config["num_clusters_kmeans_miou"],
            val_downsample_masks=val_config["val_downsample_masks"],
            patch_size=train_config["patch_size"],
            lr_heads=train_config["lr_heads"],
            gpus=_config["gpus"],
            num_classes=data_config["num_classes"],
            batch_size=train_config["batch_size"],
            num_samples=len(data_module) if num_images is None else num_images,
            pos_out_act_layer=train_config.get("pos_out_act_layer", None),
            n_pos_layers=i,
            max_epochs=train_config["max_epochs"],
            val_iters_u_segm=val_config.get("val_iters_u_segm", None),
            val_iters=val_config.get("val_iters", None),
            optimizer=train_config["optimizer"],
            exclude_norm_bias=train_config["exclude_norm_bias"],
            final_lr=train_config["final_lr"],
            weight_decay=train_config["weight_decay"],
            grad_norm_clipping=train_config.get("grad_norm_clipping", None),
            hub_repo_or_dir=train_config.get("hub_repo_or_dir", None),
            model_name=train_config.get("model_name", None),
            weights=train_config.get("weights", None),
        )

        # Create the next model's head
        if prev_model is not None:
            print(f"Loading weights from previous model for pos layer {i}.")
            # Setup the new model with the previous model's pre_pos_layers and the pos_pred layer as the
            # new pre_pos_layers and create a newly initialized pos_pred layer
            # 1) Move the pos_pred layer to the list of pre_pos_layers in the current model
            prev_model.head.pre_pos_layers.append(prev_model.head.pos_pred)
            # 2) Reinitialize the pos_pred layer from the previous model
            prev_model.head.pos_pred = model.head.pos_pred
            # 3) Load the state dict of the previous model's head to the current model's head
            msg = model.head.load_state_dict(prev_model.head.state_dict(), strict=False)
            print(
                f"Loaded Updated Previous Model Weights to a newly one for pos layer {i}:",
                msg,
            )

        checkpoint_dir = os.path.join(train_config["checkpoint_dir"], _run.experiment_info["name"])
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(checkpoint_dir, "all_ckps" + str(i)),
            save_top_k=-1,
            verbose=True,
            save_on_train_epoch_end=True,
        )
        callbacks = [checkpoint_callback]

        # Used if train data is small as for pvoc
        val_every_n_epochs = train_config.get("val_every_n_epochs")
        if val_every_n_epochs is None:
            val_every_n_epochs = 1

        # Setup trainer and start training
        trainer = Trainer(
            check_val_every_n_epoch=val_every_n_epochs,
            logger=neptune_logger,
            max_epochs=train_config["max_epochs"],
            devices=_config["gpus"],
            accelerator="cuda",
            fast_dev_run=train_config["fast_dev_run"],
            log_every_n_steps=400,
            benchmark=True,
            deterministic=False,
            num_sanity_val_steps=0,  # Disable sanity check validation
            detect_anomaly=False,  # Adds extra overhead if set to True
            callbacks=callbacks,
        )
        trainer.fit(model, datamodule=data_module, ckpt_path=py_light_chkpt)
        py_light_chkpt = None  # Reset checkpoint path to None after first iteration
        prev_model = model


def get_training_data(data_config, train_config, val_config, num_workers=12) -> TrainXVOCValDataModule:
    # Init data modules and tranforms
    data_dir = data_config["data_dir"]
    dataset_name = data_config["dataset_name"]
    input_size = data_config["size_crops"]
    # Setup data
    min_scale_factor = data_config.get("min_scale_factor", 0.25)
    max_scale_factor = data_config.get("max_scale_factor", 1.0)

    blur_strength = data_config.get("blur_strength", 1.0)
    jitter_strength = data_config.get("jitter_strength", 0.4)
    color_jitter = ColorJitter(
        0.8 * jitter_strength,
        0.8 * jitter_strength,
        0.8 * jitter_strength,
        0.2 * jitter_strength,
    )
    train_transforms = Compose(
        [
            RandomResizedCrop(
                size=(input_size, input_size),
                scale=(min_scale_factor, max_scale_factor),
            ),
            RandomApply([color_jitter], p=0.8),
            RandomGrayscale(p=0.2),
            RandomApply([GaussianBlur(sigma=[blur_strength * 0.1, blur_strength * 2.0])], p=0.5),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            RandomApply(
                [
                    RandomRotation(
                        90,
                        interpolation=InterpolationMode.NEAREST,
                        expand=False,
                        center=None,
                        fill=0,
                    )
                ],
                p=0.5,
            ),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255]),
        ]
    )

    # Setup voc dataset used for evaluation
    val_size = data_config["size_crops_val"]
    val_image_transforms = Compose(
        [
            Resize((val_size, val_size)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_target_transforms = Compose(
        [
            Resize((val_size, val_size), interpolation=InterpolationMode.NEAREST),
            ToTensor(),
        ]
    )
    val_batch_size = val_config.get("val_batch_size", train_config["batch_size"])

    val_data_module = VOCDataModule(
        batch_size=val_batch_size,
        num_workers=num_workers,
        train_split="trainaug",
        val_split="val",
        data_dir=data_config["voc_data_path"],
        train_image_transform=train_transforms,
        # val_transforms=val_image_transforms
        val_image_transform=val_image_transforms,
        val_target_transform=val_target_transforms,
    )
    num_images = None
    # Setup train data
    if dataset_name == "coco":
        num_images = None
        file_list = os.listdir(os.path.join(data_dir, "train2017"))
        train_data_module = CocoDataModule(
            batch_size=train_config["batch_size"],
            num_workers=num_workers,
            file_list=file_list,
            data_dir=data_dir,
            train_transforms=train_transforms,
            val_transforms=None,
        )
    elif dataset_name == "imagenet100":
        num_images = 126689
        with open("path/to/imagenet100.txt") as f:
            class_names = [line.rstrip("\n") for line in f]
        train_data_module = ImageNetDataModule(
            train_transforms=train_transforms,
            batch_size=train_config["batch_size"],
            class_names=class_names,
            num_workers=num_workers,
            data_dir=data_dir,
            num_images=num_images,
        )
    elif dataset_name == "imagenet1k":
        num_images = 1281167
        data_dir = os.path.join(data_dir, "train")
        class_names = os.listdir(data_dir)
        assert len(class_names) == 1000
        train_data_module = ImageNetDataModule(
            train_transforms=train_transforms,
            batch_size=train_config["batch_size"],
            class_names=class_names,
            num_workers=num_workers,
            data_dir=data_dir,
            num_images=num_images,
        )
    elif dataset_name == "voc":
        num_images = 10582
        train_data_module = VOCDataModule(
            batch_size=train_config["batch_size"],
            num_workers=num_workers,
            train_split="trainaug",
            val_split="val",
            data_dir=data_config["voc_data_path"],
            train_image_transform=train_transforms,
            val_image_transform=val_image_transforms,
            val_target_transform=val_target_transforms,
            drop_last=True,
        )
    else:
        raise ValueError(f"Data set {dataset_name} not supported")

    # Use data module wrapper to have train_data_module provide train loader and voc data module the val loader
    return TrainXVOCValDataModule(train_data_module, val_data_module), num_images


if __name__ == "__main__":
    entry_script()
