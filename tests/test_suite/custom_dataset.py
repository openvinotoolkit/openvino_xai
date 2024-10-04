import os

from torchvision import datasets


class CustomVOCDetection(datasets.VOCDetection):
    _TARGET_DIR = "Annotations"
    _TARGET_FILE_EXT = ".xml"

    def __init__(self, root, download=False, year="2012", image_set="val"):
        # Call the parent class's __init__ method
        try:
            self._SPLITS_DIR = "Main"
            super(CustomVOCDetection, self).__init__(root, year=year, image_set=image_set, download=download)
        except Exception:
            self._SPLITS_DIR = "CLS-LOC"
            voc_root = root
            self.image_set = image_set

            splits_dir = os.path.join(voc_root, "ImageSets", self._SPLITS_DIR)
            split_f = os.path.join(splits_dir, image_set.rstrip("\n") + ".txt")
            with open(os.path.join(split_f)) as f:
                file_names = [x.split()[0] for x in f.readlines()]

            image_dir = os.path.join(voc_root, "Data", self._SPLITS_DIR, self.image_set)
            self.images = [os.path.join(image_dir, x + ".JPEG") for x in file_names]

            target_dir = os.path.join(voc_root, self._TARGET_DIR, self._SPLITS_DIR, self.image_set)
            self.targets = [os.path.join(target_dir, x + self._TARGET_FILE_EXT) for x in file_names]

            assert len(self.images) == len(self.targets)
