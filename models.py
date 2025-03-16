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
    
class cxcywh_to_y1x1y2x2(torch.nn.Module):
    """
    input_shape:cxcywh(8400,4)
    outs_shape:y1x1y2x2(8400,4)
    """
    def __init__(self):
        super(cxcywh_to_y1x1y2x2, self).__init__()

    def forward(self, cxcywh):
        x1 = (cxcywh[..., 0:1] - cxcywh[..., 2:3] / 2)  # top left x
        y1 = (cxcywh[..., 1:2] - cxcywh[..., 3:4] / 2)  # top left y
        x2 = (cxcywh[..., 0:1] + cxcywh[..., 2:3] / 2)  # bottom right x
        y2 = (cxcywh[..., 1:2] + cxcywh[..., 3:4] / 2)  # bottom right y
        x1y1x2y2 = torch.cat([x1,y1,x2,y2], dim=2)
        return x1y1x2y2
