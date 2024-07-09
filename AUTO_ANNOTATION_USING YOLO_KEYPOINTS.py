from ultralytics import YOLO
import cv2
import numpy as np

class PoseEstimator:
    def __init__(self, model_path, image_path):
        self.model = YOLO(model_path)
        self.model.to('cpu')
        self.image = cv2.imread(image_path)
        self.image_height, self.image_width = self.image.shape[:2]
        self.results = self.model(self.image)
        self.upper_body_indices = [6]
        self.upper_body_indices2 = [5]
        self.lower_body_indices = [11, 12, 15, 16]
        self.lower_body_indices2 = [12]
        self.lower_body_indices3 = [11]
        self.lower_body_indices4 = [15]
        self.lower_body_indices5 = [16]
    
    def draw_bounding_box(self, bbox):
        return cv2.rectangle(self.image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    
    def draw_circles(self, coords, color):
        for coord in coords:
            cv2.circle(self.image, (int(coord[0]), int(coord[1])), 5, color, -1)
    
    def draw_lines_and_circles(self, coords, bbox, color_line, color_circle):
        points = []
        for coord in coords:
            x = bbox[0] if color_circle == (0, 255, 255) else bbox[2]
            y = int(coord[1])
            cv2.line(self.image, (int(coord[0]), int(coord[1])), (x, y), color_line, 2)
            cv2.circle(self.image, (x, y), 5, color_circle, -1)
            points.append([x, y])
        return points
    
    def find_centroid(self, coords, points):
        centroid_x = (coords[0][0] + points[0][0]) / 2
        centroid_y = (coords[0][1] + points[0][1]) / 2
        return int(centroid_x), int(centroid_y)
    
    def process(self):
        for idx, person in enumerate(self.results[0].keypoints.xy.cpu().numpy()):
            bbox = self.results[0].boxes.xyxy.cpu().numpy()[idx].astype(int)
            print("Processing person", idx + 1)
            self.draw_bounding_box(bbox)
            
            upper_body_coords = [person[idx].tolist() for idx in self.upper_body_indices]
            upper_body_coords2 = [person[idx].tolist() for idx in self.upper_body_indices2]
            lower_body_coords = [person[idx].tolist() for idx in self.lower_body_indices]
            lower_body_coords2 = [person[idx].tolist() for idx in self.lower_body_indices2]
            lower_body_coords3 = [person[idx].tolist() for idx in self.lower_body_indices3]
            lower_body_coords4 = [person[idx].tolist() for idx in self.lower_body_indices4]
            lower_body_coords5 = [person[idx].tolist() for idx in self.lower_body_indices5]
            
            self.draw_circles(upper_body_coords + lower_body_coords, (0, 0, 255))
            self.draw_circles(upper_body_coords2 + lower_body_coords, (0, 0, 255))
            
            x1_y1_points = self.draw_lines_and_circles(upper_body_coords, bbox, (255, 0, 0), (0, 255, 255))
            x2_y2_points = self.draw_lines_and_circles(upper_body_coords2, bbox, (255, 0, 0), (0, 255, 0))
            
            center = [self.find_centroid(upper_body_coords, x1_y1_points)]
            center2 = [self.find_centroid(upper_body_coords2, x2_y2_points)]
            
            self.draw_circles(lower_body_coords3, (255, 0, 255))
            
            if lower_body_coords3:
                self.draw_circles([center[0], center2[0]], (255, 255, 0))
                
                start_point = center2[0]
                end_point = self.get_end_point(start_point, lower_body_coords3)
                
                start_point2 = center[0]
                end_point2 = self.get_end_point(start_point2, lower_body_coords2)
                
                start_point3 = end_point
                end_point3 = self.get_end_point(start_point3, lower_body_coords4)
                
                start_point4 = end_point2
                end_point4 = self.get_end_point(start_point4, lower_body_coords5)
                
                if end_point and end_point2 and end_point3 and end_point4:
                    res = self.draw_min_area_rect(center[0], center2[0], end_point, end_point2, (0, 0, 255), class_id=0)
                    res2 = self.draw_min_area_rect(end_point, end_point2, end_point3, end_point4, (0, 255, 0), class_id=1)

        cv2.imwrite(r'C:\Users\intozi\Videos\pose\annotated2_image_with_centroids.jpg', self.image)
        cv2.imshow('Pose Estimation with Bounding Boxes and Centroids', self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def get_end_point(self, start_point, coords):
        for coord in coords:
            if start_point[1] < coord[1]:
                end_point = (start_point[0], int(coord[1]))
                print("End point:", end_point)
                return end_point
        return None
    
    def draw_min_area_rect(self, p1, p2, p3, p4, color, class_id):
        points = np.array([p1, p2, p3, p4])
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(self.image, [box], 0, color, 2)
        self.save_yolo_format(box, class_id)

    def save_yolo_format(self, box, class_id):
        x_coords = [point[0] for point in box]
        y_coords = [point[1] for point in box]
        x_center = np.mean(x_coords) / self.image_width
        y_center = np.mean(y_coords) / self.image_height
        width = (max(x_coords) - min(x_coords)) / self.image_width
        height = (max(y_coords) - min(y_coords)) / self.image_height       
        yolo_format = f"{class_id} {x_center} {y_center} {width} {height}\n"
        
        with open(r'C:\Users\intozi\Videos\pose\annotations.txt', 'a') as f:
            f.write(yolo_format)

def main():
    model_path = r'C:\Users\intozi\Videos\pose\yolov8x-pose-p6.pt'
    image_path = r'C:\Users\intozi\Videos\pose\0a738e4f-17de-4804-82d0-ba3ace9754d7.jpg'
    
    estimator = PoseEstimator(model_path, image_path)
    estimator.process()

if __name__ == "__main__":
    main()
