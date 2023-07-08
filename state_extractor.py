import math

CAR_INDEX = -1

class StateExtractor:

    def __init__(self) -> None:
        pass

    
    def check_overlap(self, detections, junction_threshold: list):
        # car bbox are in top left bottom right format and junction_bb contains 2 points and is a list of lists which define the threshold for oncomming traffic
        car_bboxs = [detection[0:4] for detection in detections if detection[CAR_INDEX] == 2]
        #treating junction threshold as the line. the logic will be slightly different for a rectangle
        [px1, py1], [px2, py2] = junction_threshold
        min_distance = 1e9
        at_junction = False
        
        for bbox in car_bboxs:
            # Unpack bounding box and line points
            x1, y1, x2, y2 = bbox

            if y2 > py1 or y2 > py2:
                    at_junction = True
                    continue
            
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            # Calculate the distance
            numerator = abs((py2 - py1) * center_x - (px2 - px1) * center_y + px2*py1 - py2*px1)
            denominator = math.sqrt((py2 - py1)**2 + (px2 - px1)**2)
            distance = numerator / denominator

            if distance < min_distance:
                min_distance = distance
        return at_junction, min_distance

    def check_overlap_1(self, detections, junction_bb: list):
        # car bbox are in top left bottom right format and junction_bb contains 2 points and is a list of lists which define the threshold for oncomming traffic
        car_bboxs = [detection[0:4] for detection in detections if detection[CAR_INDEX] == 2]
        [px1, py1], [px2, py2] = junction_bb
        min_distance = 1e9
        for bbox in car_bboxs:
            # Unpack bounding box and line points
            x1, y1, x2, y2 = bbox

            # Compute line equation parameters (y = mx + c)
            if px2 - px1 != 0:  # avoid division by zero
                m = (py2 - py1) / (px2 - px1)  # slope
                c = py1 - m * px1  # intercept

                # Compute y values for bounding box x coordinates
                y_top = m * x1 + c
                y_bottom = m * x2 + c

                # Calculate the minimum distance from the line to the bounding box
                if y_bottom < y1 and y_top < y1:  # if box is below the line
                    distance = y1 - max(y_bottom, y_top)
                elif y_bottom > y2 and y_top > y2:  # if box is above the line
                    distance = min(y_bottom, y_top) - y2
                else:
                    distance = 0  # if box intersects the line

                if distance < min_distance:
                    min_distance = distance
        return min_distance
            # else:
            #     return "Line is vertical"
        

if __name__ == "__main__":
    import numpy as np
    se = StateExtractor()
    res = se.check_overlap([[205, 251, 275, 296, 0.90380859375, 2.0], [285, 245, 305, 259, 0.89501953125, 2.0]], [np.array([250, 400]), np.array([400, 600])])
    print(res)
        
