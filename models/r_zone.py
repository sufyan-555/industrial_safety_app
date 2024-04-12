from ultralytics import YOLO
import cv2

class people_detection():
    """
    this class will be used to detect if people are in 
    no people zone.

    Args:
    model_path: path to model
    region: list containg regoin coordinates.
    conf: minimum confidence to consider detection 
    
    """

    def __init__(self,model_path,conf=0.45):
        """
        basic inti function
        """
        self.model=YOLO(model_path,verbose=False)
        self.conf=conf

    def in_region(self, point):
        """
        this function checks if the given point is in the region
        """
        x, y = point
        # Extracting the region points
        x1, y1 = self.region[0]
        x2, y2 = self.region[1]
        x3, y3 = self.region[2]
        x4, y4 = self.region[3]

        # Calculating vectors from point to vertices of the region
        vec1 = (x2 - x1, y2 - y1)
        vec2 = (x3 - x2, y3 - y2)
        vec3 = (x4 - x3, y4 - y3)
        vec4 = (x1 - x4, y1 - y4)

        # Calculating vectors from point to edges of the region
        edge1 = (x - x1, y - y1)
        edge2 = (x - x2, y - y2)
        edge3 = (x - x3, y - y3)
        edge4 = (x - x4, y - y4)

        # Checking if the point is on the correct side of all edges
        cross1 = vec1[0] * edge1[1] - vec1[1] * edge1[0]
        cross2 = vec2[0] * edge2[1] - vec2[1] * edge2[0]
        cross3 = vec3[0] * edge3[1] - vec3[1] * edge3[0]
        cross4 = vec4[0] * edge4[1] - vec4[1] * edge4[0]

        # If all cross products have the same sign, the point is inside the region
        return ((cross1 >= 0 and cross2 >= 0 and cross3 >= 0 and cross4 >= 0) or 
               (cross1 <= 0 and cross2 <= 0 and cross3 <= 0 and cross4 <= 0))

    
    def process(self,img,region=False,flag=True):
        """
        this function processes the cv2 frame and returns the
        bounding boxes
        """
        self.region=region
        if not flag:
            return (False,[])

        bb_boxes=[]
        results=self.model.track(img,verbose=False)

        for box in results[0].boxes:
            if (int(box.cls[0])==0 and float(box.conf[0])>self.conf):
                bb=list(map(int,box.xyxy[0]))
                
                if self.region:
                    center=[(bb[0]+bb[2])//2,(bb[1]+bb[3])//2]

                    if(self.in_region(center)):
                        bb_boxes.append(bb)
                else:
                    bb_boxes.append(bb)
        
        if(len(bb_boxes)):
            found=True
        else:
            found=False
        return (found,bb_boxes)