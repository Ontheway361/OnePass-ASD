# -*- coding: utf-8 -*-

def calculate_iou(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = max(bbox1[3], bbox2[3])
    inter = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    area_a = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
    area_b = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)
    iou = inter / (area_a + area_b - inter + 1e-3)
    return iou