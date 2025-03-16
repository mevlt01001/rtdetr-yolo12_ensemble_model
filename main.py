from torch_to_onnx import model_to_onnx
from snc4onnx import combine
from models import cxcywh_and_scores
from models import cxcywh_to_x1y1x2y2


if __name__ == "__main__":
    model_to_onnx(
        model=cxcywh_and_scores(),
        onnx_name="cxcywh_and_scores.onnx",
        input_shape=(1,84,8400),
        OPSET=13,
        input_names=["input"],
        output_names=["boxes_cxcywh", "person_conf"]
    )

    model_to_onnx(
        model=cxcywh_to_x1y1x2y2(),
        onnx_name="cxcywh_to_x1y1x2y2.onnx",
        input_shape=(8400,4),
        OPSET=13,
        input_names=["cxcywh"],
        output_names=["x1y1x2y2"]
    )

    combine(
        input_onnx_file_paths=["onnx_folder/cxcywh_and_scores.onnx", "onnx_folder/cxcywh_to_x1y1x2y2.onnx"],
        srcop_destop=[["boxes_cxcywh", "cxcywh"]],
        op_prefixes_after_merging=["init", "next"],
        output_onnx_file_path="onnx_folder/person_conf_and_x1y1x2y2.onnx"
    )