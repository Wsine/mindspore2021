#!/usr/bin/env python
class Bbox:
    
    def __init__(
        self, xmin, ymin, xmax, ymax, 
        p0, p1, p2, p3, image_id
    ):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.image_id = image_id
        
    def intersect_area(self, other_bbox):
        xdiff = min(self.xmax, other_bbox.xmax) - max(self.xmin, other_bbox.xmin)
        ydiff = min(self.ymax, other_bbox.ymax) - max(self.ymin, other_bbox.ymin)
        if xdiff < 0 or ydiff < 0:
            return 0
        else:
            if xdiff < 1e-6:
                xdiff = 1e-6
            if ydiff < 1e-6:
                ydiff = 1e-6
            return xdiff * ydiff
    
    def area(self):
        a = (self.xmax - self.xmin) * (self.ymax - self.ymin) 
        if a < 1e-6:
            return 1e-6
        else:
            return a
        
    def __getitem__(self, key):
        if key == 'p0':
            return self.p0
        elif key == 'p1':
            return self.p1
        elif key == 'p2':
            return self.p2
        elif key == 'p3':
            return self.p3
        
    def __repr__(self):
        return f"xmin: {self.xmin}, ymin: {self.ymin}, xmax: {self.xmax}, ymax: {self.ymax}, p0: {self.p0}, p1: {self.p1}, p2: {self.p2}, p3: {self.p3}, image_id: {self.image_id}\n\n"
    
            
output_columns = {'xmin', 'ymin', 'xmax', 'ymax', 'p0', 'p1', 'p2', 'p3', 'image_id'}

def read_participants_input(df):
    assert output_columns.issubset(df.columns), f'Output format wrong, missing column {output_columns - set(df.columns)}'
    prediction_bboxes = [
        Bbox(
            item.xmin,
            item.ymin,
            item.xmax,
            item.ymax,
            item.p0,
            item.p1,
            item.p2,
            item.p3,
            item.image_id
        )
        for i, item in df.iterrows()
    ]
    return prediction_bboxes