import argparse
import os

from torcheval.metrics.functional import binary_auprc

from src.deep_ad.config import Config
from src.deep_ad.data.dagm_dataset import DAGMDataset
from src.deep_ad.eval import (
    compute_heatmap,
    load_pretrained,
    cut_margins,
    reconstruct_by_inpainting,
    diff_postprocessing,
    remove_islands,
)
from src.deep_ad.image import plot_images
from src.deep_ad.measurements import Stopwatch
from src.deep_ad.save_manager import SaveManager
from src.deep_ad.transforms import create_test_transform


def main() -> None:
    # Load the pretrained configuration
    config = Config(root_dir=ARGS.root_dir, config_path=ARGS.config_path)
    save_manager = SaveManager(config)
    pretrained_run_name, pretrained_checkpoint_name = ARGS.pretrained.split("/")
    print(f"\n\nLoading model from run {pretrained_run_name}/{pretrained_checkpoint_name}.")

    if ARGS.device:
        assert ARGS.device == "cpu" or ARGS.device == "cuda"
        config.device = ARGS.device

    detection_classes = list(range(1, 11))
    if ARGS.detection_classes:
        if "-" in ARGS.detection_classes:
            start, end = ARGS.detection_classes.split("-")
            detection_classes = list(range(int(start), int(end) + 1))
        else:
            detection_classes = [int(ARGS.detection_classes)]
        assert all(1 <= c <= 10 for c in detection_classes), "Classes must be between 1 and 10."

    limit_images = float("inf")
    if ARGS.limit_images:
        limit_images = ARGS.limit_images

    print(f"\nRunning training with the following configuration:\n\n{config}")

    # Check if existing detections can be loaded
    detections_dir = save_manager.get_detections_dir(
        config.save_dir, run_name=pretrained_run_name, checkpoint_name=pretrained_checkpoint_name
    )
    if not os.path.exists(detections_dir):
        os.makedirs(detections_dir)
    print(f"\n\nDetections will be saved to '{detections_dir}'")

    if not ARGS.yes:
        proceed = input("\n\nProceed with detection? (y/n): ")
        if proceed.lower() != "y":
            print("Detection aborted.")
            return
    print("\n\nDetection started.\n")

    # TODO load existing detections if found

    # Load pretrained model
    model = load_pretrained(
        config, save_manager, run_name=pretrained_run_name, checkpoint_name=pretrained_checkpoint_name
    )
    model = model.to(config.device)

    # Load the data
    test_transform = create_test_transform()
    test_dataset = DAGMDataset(
        img_dir=config.DAGM_raw_dir,
        transform=test_transform,
        target_transform=test_transform,
        classes=[10],
        type="Defect-only",
        train=False,
    )

    # Iterate through images
    stopwatch = Stopwatch()
    stopwatch.start()
    for idx, (image, image_mask) in enumerate(test_dataset):
        stopwatch.checkpoint()
        if idx >= limit_images:
            break

        image, image_mask = image.squeeze(), image_mask.squeeze()
        inpainted_image = reconstruct_by_inpainting(config, image, model)
        diff_image = diff_postprocessing(image - inpainted_image)
        diff_cut = cut_margins(diff_image.clone(), margin=1)

        # Break diff image into patches and compute a heatmap by applying a metric over each patch
        patch_metric = lambda patch: patch.max()
        hm_num_windows = (diff_image.shape[-1] - config.hm_patch_size) // config.hm_patch_size + 1
        heatmap = compute_heatmap(diff_cut, config.hm_patch_size, hm_num_windows, patch_metric)

        # Binarize heatmap and remove islands
        heatmap[heatmap < config.hm_threshold] = 0
        heatmap[heatmap >= config.hm_threshold] = 1
        heatmap = remove_islands(heatmap, patch_size=config.hm_patch_size)
        diff_image = diff_image * heatmap

        # Compute final metrics
        auprc = binary_auprc(diff_image.reshape(-1), image_mask.reshape(-1))
        print(
            f"Image {idx + 1:3d}/{min(limit_images, len(test_dataset))}: AUPRC={auprc:.6f} time={stopwatch.elapsed_since_last_checkpoint():.3f}s"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use pretrained model for detecting visual anomalies.")

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
    parser.add_argument(
        "--pretrained", dest="pretrained", type=str, required=True, help="Checkpoint id: <run_name>/<checkpoint_name>"
    )
    parser.add_argument("-y", "--yes", dest="yes", action="store_true", help="Proceed without confirmation.")

    # Prediction arguments
    parser.add_argument("--device", dest="device", type=str, required=True, help="Either cpu or cuda.")
    parser.add_argument(
        "--detection-classes",
        dest="detection_classes",
        type=str,
        required=True,
        help="Which classes to run the detection on. Can be passed as a single number, or as a range: <start>-<end (inclusive)>",
    )
    parser.add_argument(
        "-l", "--limit", dest="limit_images", type=int, required=False, help="Limit the number of images for detection."
    )

    ARGS = parser.parse_args()
    main()
