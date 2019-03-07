import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image

from pycocotools.coco import COCO


class CocoCaptionDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, json, file_name=False):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            json: coco annotation file path.
        """
        self.coco = COCO(json)
        self.ids = sorted(self.coco.anns.keys())
        self.file_name = file_name


    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        coco = self.coco
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        image = coco.imgs[coco.anns[ann_id]['image_id']]['file_name']

        if self.file_name:
            return caption, image
        return caption

    def __len__(self):
        return len(self.ids)


if __name__ == '__main__':
    # dataset = CocoCaptionDataset('/data/vision/torralba/datasets/COCO/annotations/captions_train2014.json')
    # with open('coco_captions.txt', 'w') as f:
        # for caption in dataset:
            # caption = caption.strip()
            # if caption:
                # f.write('{}\n'.format(caption))

    dataset = CocoCaptionDataset('/data/vision/torralba/datasets/COCO/annotations/captions_val2014.json')
    with open('coco_captions_val.txt', 'w') as f:
        for caption in dataset:
            caption = caption.strip()
            if caption:
                f.write('{}\n'.format(caption))
