import torch

class cxcywh_and_scores(torch.nn.Module):
    """
    `input_shape`:YOLO12_raw_out(1,84,8400)\n
    `outs_shape`:cxcxywh(1,8400,4),person_conf(1,1,8400)
    """
    def __init__(self):
        super(cxcywh_and_scores, self).__init__()

    def forward(self, x):
        # x.shape is (1,84,8400)
        x = torch.permute(x, (0, 2, 1)) # x.shape is (1,8400,84)
        boxes = x[..., :4] # cxcywh [1, 8400, 4]
        person_conf = x[..., 4:5] # [1, 8400, 1]
        person_conf = torch.permute(person_conf, (0, 2, 1)) # person_conf.shape is (1, 1, 8400)
        return boxes, person_conf
    
class cxcywh_to_x1y1x2y2(torch.nn.Module):
    """
    input_shape:cxcywh(1,8400,4)
    outs_shape:x1y1x2y2(1,8400,4)
    """
    def __init__(self):
        super(cxcywh_to_x1y1x2y2, self).__init__()

    def forward(self, cxcywh):
        x1 = (cxcywh[..., 0:1] - cxcywh[..., 2:3] / 2)  # top left x
        y1 = (cxcywh[..., 1:2] - cxcywh[..., 3:4] / 2)  # top left y
        x2 = (cxcywh[..., 0:1] + cxcywh[..., 2:3] / 2)  # bottom right x
        y2 = (cxcywh[..., 1:2] + cxcywh[..., 3:4] / 2)  # bottom right y
        x1y1x2y2 = torch.cat([x1,y1,x2,y2], dim=1)
        return x1y1x2y2

class cxcywh_to_y1x1y2x2(torch.nn.Module):
    """
    input_shape:cxcywh(1,8400,4)
    outs_shape:y1x1y2x2(1,8400,4)
    """
    def __init__(self):
        super(cxcywh_to_y1x1y2x2, self).__init__()

    def forward(self, cxcywh):
        y1 = (cxcywh[..., 1:2] - cxcywh[..., 3:4] / 2)  # top left y
        x1 = (cxcywh[..., 0:1] - cxcywh[..., 2:3] / 2)  # top left x
        y2 = (cxcywh[..., 1:2] + cxcywh[..., 3:4] / 2)  # bottom right y
        x2 = (cxcywh[..., 0:1] + cxcywh[..., 2:3] / 2)  # bottom right x
        y1x1y2x2 = torch.cat([y1,x1,y2,x2], dim=2)
        return y1x1y2x2