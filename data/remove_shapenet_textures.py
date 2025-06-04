from pathlib import Path
import shutil
import csv
import logging
import argparse

from dataset.shapenet import AVAILABLE_CATEGORIES
from utils.logging import get_predefined_logger


def process_category(category_path: Path) -> None:
    for directory in category_path.iterdir():
        if not directory.is_dir():
            continue
        full_ID = directory.stem
        texture_pathes = [category_path / full_ID / 'images', category_path / full_ID / full_ID]
        for texture_path in texture_pathes:
            if texture_path.exists():
                logger.info(f'remove {texture_path}')
                shutil.rmtree(texture_path)

def main():
    logger = get_predefined_logger(__name__)
    logger.setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('shapenet_path', help='Path to the ShapeNetCore.v1 dataset')
    config = parser.parse_args()
    dataset_path = Path(config.shapenet_path)

    for category_ID in AVAILABLE_CATEGORIES.values():
        category_path = dataset_path / category_ID
        process_category(category_path)

    logging.shutdown()

if __name__ == '__main__':
    main()
