import os
import numpy as np
from torch_to_onnx import model_to_onnx
from snc4onnx import combine
from sog4onnx import generate
from soc4onnx import change
from models import cxcywh_and_scores
from models import cxcywh_to_y1x1y2x2


if __name__ == "__main__":
    """
    YOLO12 MODELİN ÇIKIŞI (1,84,8400)
    """
    
    # input: YOLO12_raw_out(1,84,8400)
    # output: cxcywh(1,8400,4), person_conf(1,1,8400)-type:int64
    model_to_onnx(
        model=cxcywh_and_scores(),
        onnx_name="cxcywh_and_scores.onnx",
        input_shape=(1,84,8400),
        OPSET=19,
        input_names=["YOLO_raw_out"],
        output_names=["raw_cxcywh", "raw_person_conf"]
    )

    # input: cxcywh(1,8400,4)
    # output: y1x1y2x2(1,8400,4)
    model_to_onnx(
        model=cxcywh_to_y1x1y2x2(),
        onnx_name="cxcywh_to_y1x1y2x2.onnx",
        input_shape=(1,8400,4),
        OPSET=19,
        input_names=["cxcywh"],
        output_names=["y1x1y2x2"]
    )

    # input: YOLO12_raw_out(1,84,8400)
    # output: y1x1y2x2(1,8400,4), person_conf(1,1,8400)
    combine(
        input_onnx_file_paths=[
            "onnx_folder/cxcywh_and_scores.onnx",
            "onnx_folder/cxcywh_to_y1x1y2x2.onnx"
        ],
        srcop_destop=[
            ["raw_cxcywh", "cxcywh"],
        ],
        op_prefixes_after_merging=[
            "cxcywh_and_scores", "cxcywh_to_y1x1y2x2"
        ],
        output_onnx_file_path="onnx_folder/y1x1y2x2_and_scores.onnx",
    )

    os.remove("onnx_folder/cxcywh_and_scores.onnx")
    os.remove("onnx_folder/cxcywh_to_y1x1y2x2.onnx")

    # max output boxes
    generate(
        op_type='Constant',
        opset=19,
        op_name='max_output_boxes',
        output_variables={"max_output_boxes": [np.int64, [1]]},
        attributes={"value": np.array([25], dtype=np.int64)},  # NumPy array olmalı
        output_onnx_file_path="onnx_folder/max_output_boxes.onnx",
    )

    # iou threshold
    generate(
        op_type='Constant',
        opset=19,
        op_name='iou_threshold',
        output_variables={"iou_threshold": [np.float32, [1]]},
        attributes={"value": np.array([0.45], dtype=np.float32)},  # NumPy array olmalı
        output_onnx_file_path="onnx_folder/iou_threshold.onnx",
    )

    # score threshold
    generate(
        op_type='Constant',
        opset=19,
        op_name='score_threshold',
        output_variables={"score_threshold": [np.float32, [1]]},
        attributes={"value": np.array([0.5], dtype=np.float32)},  # NumPy array olmalı
        output_onnx_file_path="onnx_folder/score_threshold.onnx",
    )

    # NMS
    generate(
        op_type='NonMaxSuppression',
        opset=19,
        op_name='NonMaxSuppression',
        input_variables={
            "nms_boxes": [np.float32, [1,8400, 4]], 
            "nms_scores": [np.float32, [1,1,8400]],
            "nms_max_output_boxes": [np.int64, [1]],
            "nms_iou_threshold": [np.float32, [1]],
            "nms_score_threshold": [np.float32, [1]],
        },
        output_variables={
            "selected_indices": [np.int64, ['N',3]]
        },
        attributes={
            "center_point_box": 0,
        },
        output_onnx_file_path="onnx_folder/NMS.onnx",
    )

    combine(
        input_onnx_file_paths=[
            "onnx_folder/max_output_boxes.onnx",
            "onnx_folder/NMS.onnx"
        ],
        srcop_destop=[
            ["max_output_boxes", "nms_max_output_boxes"],
        ],
        output_onnx_file_path="onnx_folder/NMS.onnx",
    )
    os.remove("onnx_folder/max_output_boxes.onnx")

    combine(
        input_onnx_file_paths=[
            "onnx_folder/iou_threshold.onnx",
            "onnx_folder/NMS.onnx"
        ],
        srcop_destop=[
            ["iou_threshold", "nms_iou_threshold"],
        ],
        output_onnx_file_path="onnx_folder/NMS.onnx",
    )
    os.remove("onnx_folder/iou_threshold.onnx")

    combine(
        input_onnx_file_paths=[
            "onnx_folder/score_threshold.onnx",
            "onnx_folder/NMS.onnx"
        ],
        srcop_destop=[
            ["score_threshold", "nms_score_threshold"],
        ],
        output_onnx_file_path="onnx_folder/NMS.onnx",
    )
    os.remove("onnx_folder/score_threshold.onnx")

    change(
        input_onnx_file_path="onnx_folder/NMS.onnx",
        output_onnx_file_path="onnx_folder/NMS.onnx",
        opset=19
    )
