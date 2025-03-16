import onnx
from onnxsim import simplify
import torch, os
from cxcywh_and_scores import cxcywh_and_scores

onnx_folder_path = 'onnx_folder'
os.makedirs(onnx_folder_path, exist_ok=True)

def model_to_onnx(model: torch.nn.Module,
                  onnx_name: str, 
                  input_shape: tuple, 
                  OPSET: int, 
                  input_names: list, 
                  output_names: list):

    onnx_file_path = os.path.join(onnx_folder_path, onnx_name)
    x = torch.randn(input_shape)

    torch.onnx.export(
        model,
        args=(x),
        f=onnx_file_path,
        opset_version=OPSET,
        input_names=input_names,
        output_names=output_names,
    )
    model_onnx1 = onnx.load(onnx_file_path)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, onnx_file_path)

    model_onnx2 = onnx.load(onnx_file_path)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file_path)
    


cxcywh_and_scores_model = cxcywh_and_scores()

model_to_onnx(
    model=cxcywh_and_scores_model,
    onnx_name='cxcywh_and_scores.onnx',
    input_shape=(1,84,8400),
    OPSET=19,
    input_names=['predictions'],
    output_names=['boxes_cxcywh', 'person_scores']
)

