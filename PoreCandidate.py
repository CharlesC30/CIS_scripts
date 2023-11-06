from collections import OrderedDict

class PoreCandidate:
    def __init__(self, t0, bbox0):
        self.exists = True
        self.bboxs = OrderedDict()
        self.bboxs[t0] = bbox0
    
    def update_bbox(self, t, bbox):
        self.bboxs[t] = bbox

    def get_current_bbox(self):
        current_bbox = list(self.bboxs.values())[-1]
        return current_bbox
    
    def end(self):
        self.exists = False

    @property
    def min_threshold(self):
        return list(self.bboxs.keys())[0]
    
    @property
    def max_threshold(self):
        return list(self.bboxs.keys())[-1]
