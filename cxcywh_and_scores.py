import torch

class cxcywh_and_scores(torch.nn.Module):
    """
    `input_shape`:YOLO12_raw_out(1,84,8400)
    `outs_shape`:cxcxywh(8400,4),person_conf(8400,1)
    """
    def __init__(self):
        super(cxcywh_and_scores, self).__init__()

    def forward(self, x):
        # x.shape is (1,84,8400)
        x = torch.t(x[0]) # x.shape is (8400,84)
        boxes = x[..., :4] # cxcywh [boxes, 4]
        person_conf = x[..., 4:5] # [boxes, 1]
        person_conf = torch.sqrt(person_conf)
        return boxes, person_conf
    
