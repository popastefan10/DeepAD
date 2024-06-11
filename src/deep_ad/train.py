import argparse
import os
import torch

from torch.utils.data import DataLoader

from src.deep_ad.config import Config
from src.deep_ad.data.dagm_split import dagm_patch_get_splits
from src.deep_ad.image import plot_losses
from src.deep_ad.model import DeepCNN
from src.deep_ad.save_manager import SaveManager
from src.deep_ad.trainer import PretrainedDict, Trainer, create_optimizer
from src.deep_ad.transforms import create_training_transform, create_validation_transform


def main() -> None:
    # Load the configuration
    config = Config(root_dir=ARGS.root_dir, config_path=ARGS.config_path)
    if ARGS.pretrained:
        pretrained_run_name, pretrained_checkpoint_name = ARGS.pretrained.split("/")
        print(f"\n\nLoading model from run {pretrained_run_name}/{pretrained_checkpoint_name}.")
        config.load(SaveManager.get_config_path(config.save_dir, run_name=pretrained_run_name))
    if ARGS.device:
        assert ARGS.device == "cpu" or ARGS.device == "cuda"
        config.device = ARGS.device
    if ARGS.epochs:
        config.train_epochs = ARGS.epochs
    if ARGS.train_classes:
        fst = ARGS.train_classes[0]
        if len(ARGS.train_classes) == 1 and fst >= 100:
            config.train_classes = list(range(int(fst // 100), int(fst % 100)))
        else:
            config.train_classes = ARGS.train_classes
        assert all(1 <= c <= 10 for c in config.train_classes), "Classes must be between 1 and 10."
    print(f"\nRunning training with the following configuration:\n\n{config}")

    run_name = ARGS.run_name
    print(f"\n\nResults will be saved to '{run_name}'.")

    if not ARGS.yes:
        proceed = input("\n\nProceed with training? (y/n): ")
        if proceed.lower() != "y":
            print("Training aborted.")
            return
    print("\n\nTraining started.\n")

    # Load the data
    train_transform = create_training_transform(config)
    val_transform = create_validation_transform(config)
    train_dataset, val_dataset = dagm_patch_get_splits(
        config, train_transform, val_transform, classes=config.train_classes, cache_patches=ARGS.cache_patches
    )
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    # test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # Load the model
    model = DeepCNN(config).to(config.device)
    optimizer = create_optimizer(model, config)
    model_num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {model_num_params:,} parameters")
    pretrained_dict: PretrainedDict | None = None

    if ARGS.pretrained:
        checkpoint_path = SaveManager.get_checkpoint_path(
            config, run_name=pretrained_run_name, checkpoint_name=pretrained_checkpoint_name
        )
        model, optimizer, train_losses, val_losses, epoch = SaveManager.load_checkpoint(
            model,
            optimizer,
            path=checkpoint_path,
        )
        pretrained_dict = {"epoch": epoch, "train_losses": train_losses, "val_losses": val_losses}

    # Create the trainer
    trainer = Trainer(
        config,
        model,
        optimizer,
        train_dataloader,
        val_dataloader,
        run_name=run_name,
        train_epochs=config.train_epochs,
        limit_batches=ARGS.limit_batches,
        shuffle_data=ARGS.shuffle_data,
        save_epochs=ARGS.save_epochs,
        pretrained_dict=pretrained_dict,
    )

    # Start training
    train_losses, val_losses = trainer.train(plot_period=ARGS.plot_period, plot_train=True, plot_val=False)
    plot_dir = f"{ARGS.root_dir}/save/plots/{run_name}"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plot_losses(
        torch.asarray(train_losses),
        torch.asarray(val_losses),
        title=f"Train for {config.train_epochs} epochs on one batch from Class 10",
        save_path=f"{plot_dir}/losses.png",
        show_plot=False,
    )

    # Clear cache after training
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model for visual anomaly detection.")

    # Required arguments
    parser.add_argument(
        "-r",
        "--root",
        dest="root_dir",
        type=str,
        required=True,
        help="Root directory of the project, containing data, models, notebooks, save and src as subdirectories.",
    )
    parser.add_argument(
        "-c", "--config", dest="config_path", type=str, required=True, help="Path to the configuration file."
    )
    parser.add_argument("-n", "--name", dest="run_name", type=str, required=True, help="Name of the run.")
    parser.add_argument("-y", "--yes", dest="yes", action="store_true", help="Proceed without confirmation.")

    # Training arguments
    parser.add_argument("--device", dest="device", type=str, required=True, help="Either cpu or cuda.")
    parser.add_argument("-e", "--epochs", dest="epochs", type=int, required=True, help="Number of epochs to train for.")
    parser.add_argument(
        "--train-classes",
        dest="train_classes",
        type=int,
        nargs="+",
        required=True,
        help="Which classes to train on. Pass a list of numbers, or 110 for all classes.",
    )
    parser.add_argument(
        "-l", "--limit", dest="limit_batches", type=int, required=False, help="Limit the number of batches per epoch."
    )
    parser.add_argument(
        "--no-shuffle",
        dest="shuffle_data",
        action="store_false",
        required=False,
        help="Whether or not to shuffle the data during training.",
    )
    parser.set_defaults(shuffle_data=True)
    parser.add_argument(
        "--plot-period",
        dest="plot_period",
        type=int,
        required=True,
        help="The fixed number of epochs after we repeatedly plot the outputs (e.g. plot once every 10 epochs).",
    )
    parser.add_argument("--save-epochs", dest="save_epochs", type=int, nargs="+", help="Epochs to save the model at.")
    parser.add_argument("--pretrained", dest="pretrained", type=str, help="Checkpoint id: <run_name>/<checkpoint_name>")
    parser.add_argument(
        "--cache-patches", dest="cache_patches", action="store_true", help="Cache patches for faster training."
    )
    parser.set_defaults(cache_patches=False)

    ARGS = parser.parse_args()
    main()
