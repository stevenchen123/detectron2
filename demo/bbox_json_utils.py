from typing import Sequence
from torch import Tensor
from detectron2.structures import Boxes
import json
import numpy as np


class BBoxBuilder:
    def __init__(self):
        self._data = []

    def add_frame_prediction(
        self,
        frame_id: int,
        pred_class: Sequence[str],
        scores: Tensor,
        bbox: Boxes,
        image_shape: np.ndarray,
    ) -> None:
        n = len(pred_class)
        bbox = [b.numpy() for b in bbox]
        scores = scores.numpy().astype(np.float64)
        frame_pred = {}
        frame_pred["frame_id"] = frame_id
        # note dimension is flipped
        frame_pred["image_shape"] = [image_shape[1], image_shape[0]]
        frame_pred["instances"] = []
        for i in range(n):
            c = pred_class[i]
            s = scores[i]
            bbox_row = bbox[i]
            bbox_row = [int(x) for x in bbox_row]
            bbox_row[2] = bbox_row[2] - bbox_row[0]
            bbox_row[3] = bbox_row[3] - bbox_row[1]
            frame_pred["instances"].append(
                {"class": c, "score": s, "x,y,dx,dy": bbox_row}
            )
        self._data.append(frame_pred)
        return

    def save(self, filename: str) -> None:
        if not filename.endswith("json"):
            raise ValueError("unexpected type")
        with open(filename, "w") as f:
            json.dump(self._data, f, indent=4)
