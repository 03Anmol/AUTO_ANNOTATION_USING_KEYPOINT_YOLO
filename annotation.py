import os
import cv2
import numpy as np
from ultralytics import YOLO

class PoseEstimator:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.model.to('cpu')
        self.upper_body_indices = [6]
        self.upper_body_indices2 = [5]
        self.lower_body_indices = [11, 12, 15, 16]
        self.lower_body_indices2 = [12]
        self.lower_body_indices3 = [11]
        self.lower_body_indices4 = [15]
        self.lower_body_indices5 = [16]

    def process_image(self, image_path):
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error reading image: {image_path}")
                return

            self.image_height, self.image_width = image.shape[:2]
            results = self.model(image)
            image_filename = os.path.splitext(os.path.basename(image_path))[0]

            total_persons = len(results[0].keypoints.xy.cpu().numpy())
            print(f"Total persons detected: {total_persons}")

            for idx, person in enumerate(results[0].keypoints.xy.cpu().numpy()):
                try:
                    bbox = results[0].boxes.xyxy.cpu().numpy()[idx].astype(int)
                    person_number = idx + 1
                    print(f"Processing Person {person_number}")
                    roi = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    roi_height, roi_width = roi.shape[:2]
                    # self.draw_bounding_box(roi, [0, 0, roi_width, roi_height], person_number)
                    roi_path = f'C:\\Users\\intozi\\Videos\\pose\\dataset\\images\\{image_filename}_person_{person_number}.jpg'
                    cv2.imwrite(roi_path, roi)
                    print(f"Saved annotated ROI for Person {person_number}: {roi_path}")

                    adjusted_person = []
                    for point in person:
                        x_adjusted = (point[0] - bbox[0]) * (roi_width / (bbox[2] - bbox[0]))
                        y_adjusted = (point[1] - bbox[1]) * (roi_height / (bbox[3] - bbox[1]))
                        adjusted_person.append([x_adjusted, y_adjusted])
                    adjusted_person = np.array(adjusted_person)

                    upper_body_coords = [adjusted_person[idx].tolist() for idx in self.upper_body_indices]
                    upper_body_coords2 = [adjusted_person[idx].tolist() for idx in self.upper_body_indices2]
                    lower_body_coords = [adjusted_person[idx].tolist() for idx in self.lower_body_indices]
                    lower_body_coords2 = [adjusted_person[idx].tolist() for idx in self.lower_body_indices2]
                    lower_body_coords3 = [adjusted_person[idx].tolist() for idx in self.lower_body_indices3]
                    lower_body_coords4 = [adjusted_person[idx].tolist() for idx in self.lower_body_indices4]
                    lower_body_coords5 = [adjusted_person[idx].tolist() for idx in self.lower_body_indices5]

                    # self.draw_circles(roi, upper_body_coords + lower_body_coords, (0, 0, 255))
                    # self.draw_circles(roi, upper_body_coords2 + lower_body_coords, (0, 0, 255))

                    x1_y1_points = self.draw_lines_and_circles(roi, upper_body_coords, [0, 0, roi_width, roi_height], (255, 0, 0), (0, 255, 255))
                    x2_y2_points = self.draw_lines_and_circles(roi, upper_body_coords2, [0, 0, roi_width, roi_height], (255, 0, 0), (0, 255, 0))

                    center = [self.find_centroid(upper_body_coords, x1_y1_points)]
                    center2 = [self.find_centroid(upper_body_coords2, x2_y2_points)]

                    #self.draw_circles(roi, lower_body_coords3, (255, 0, 255))

                    if lower_body_coords3:
                        #self.draw_circles(roi, [center[0], center2[0]], (255, 255, 0))

                        start_point = center2[0]
                        end_point = self.get_end_point(start_point, lower_body_coords3)

                        start_point2 = center[0]
                        end_point2 = self.get_end_point(start_point2, lower_body_coords2)

                        start_point3 = end_point
                        end_point3 = self.get_end_point(start_point3, lower_body_coords4)

                        start_point4 = end_point2
                        end_point4 = self.get_end_point(start_point4, lower_body_coords5)

                        if end_point and end_point2 and end_point3 and end_point4:
                            self.draw_min_area_rect(roi, center[0], center2[0], end_point, end_point2, (0, 0, 255), person_number, class_id=0, image_filename=image_filename)
                            self.draw_min_area_rect(roi, end_point, end_point2, end_point3, end_point4, (0, 255, 0), person_number, class_id=1, image_filename=image_filename)

                except Exception as e:
                    print(f"Error processing person {person_number} in {image_filename}: {e}")

            # annotated_image_path = f'C:\\Users\intozi\\Videos\\pose\\dataset\\images\\{image_filename}.jpg'
            # cv2.imwrite(annotated_image_path, image)
            # print(f"Saved annotated image: {annotated_image_path}")

        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

    def draw_bounding_box(self, image, bbox, person_number):
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(image, f'Person {person_number}', (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    def draw_circles(self, image, coords, color):
        for coord in coords:
            cv2.circle(image, (int(coord[0]), int(coord[1])), 5, color, -1)

    def draw_lines_and_circles(self, image, coords, bbox, color_line, color_circle):
        points = []
        for coord in coords:
            x = bbox[0] if color_circle == (0, 255, 255) else bbox[2]
            y = int(coord[1])
            #cv2.line(image, (int(coord[0]), int(coord[1])), (x, y), color_line, 2)
            #cv2.circle(image, (x, y), 5, color_circle, -1)
            points.append([x, y])
        return points

    def find_centroid(self, coords, points):
        centroid_x = (coords[0][0] + points[0][0]) / 2
        centroid_y = (coords[0][1] + points[0][1]) / 2
        return int(centroid_x), int(centroid_y)

    def get_end_point(self, start_point, coords):
        for coord in coords:
            if start_point[1] < coord[1]:
                end_point = (start_point[0], int(coord[1]))
                #print("End point:", end_point)
                return end_point
        return None

    def draw_min_area_rect(self, image, p1, p2, p3, p4, color, person_number, class_id, image_filename):
        points = np.array([p1, p2, p3, p4])
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        #cv2.drawContours(image, [box], 0, color, 2)
        self.save_yolo_format(image, box, person_number, class_id, image_filename)

    def save_yolo_format(self, image, box, person_number, class_id, image_filename):
        image_height, image_width = image.shape[:2]
        x_coords = [point[0] for point in box]
        y_coords = [point[1] for point in box]
        x_center = np.mean(x_coords) / image_width
        y_center = np.mean(y_coords) / image_height
        width = (max(x_coords) - min(x_coords)) / image_width
        height = (max(y_coords) - min(y_coords)) / image_height
        yolo_format = f"{class_id} {x_center} {y_center} {width} {height}\n"

        output_directory = r'C:\Users\intozi\Videos\pose\dataset\labels'
        os.makedirs(output_directory, exist_ok=True)
        file_path = os.path.join(output_directory, f'{image_filename}_person_{person_number}.txt')

        with open(file_path, 'a') as f:
            f.write(yolo_format)

def main():
    model_path = r'C:\Users\intozi\Videos\pose\yolov8l-pose.pt'
    folder_path = r'C:\Users\intozi\Videos\pose\dataset\people_images'
    estimator = PoseEstimator(model_path)
    
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, filename)
            print(f"Processing image: {image_path}")
            estimator.process_image(image_path)

if __name__ == "__main__":
    main()
