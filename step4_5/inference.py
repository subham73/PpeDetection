import cv2
import os
import argparse
from ultralytics import YOLO

# Define label mappings
person_label_mapping = {0: 'person'}
ppe_label_mapping = {0: 'hard-hat', 1: 'gloves', 2: 'boots', 3: 'vest', 4: 'ppe-suit'}

def draw_boxes(image, boxes, labels, confidences, label_mapping, confidence_threshold=0.8):
    for (box, label, confidence) in zip(boxes, labels, confidences):
        if confidence >= confidence_threshold:
            x_min, y_min, x_max, y_max = box
            label_text = label_mapping.get(label, 'unknown')
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            # Decrease font size and place label inside the bounding box
            cv2.putText(image, f'{label_text}', (x_min + 5, y_min + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def process_image(image_path, output_path, person_model, ppe_model):
    # Read image
    image = cv2.imread(image_path)

    # Person detection
    person_results = person_model.predict(image, visualize=False)
    for result in person_results:
        person_boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        person_labels = result.boxes.cls.cpu().numpy().astype(int)
        person_confidences = result.boxes.conf.cpu().numpy()

        # Crop and PPE detection
        for box in person_boxes:
            x_min, y_min, x_max, y_max = box
            cropped_image = image[y_min:y_max, x_min:x_max]
            ppe_results = ppe_model.predict(cropped_image, visualize=False)
            
            for ppe_result in ppe_results:
                ppe_boxes = ppe_result.boxes.xyxy.cpu().numpy().astype(int)
                ppe_labels = ppe_result.boxes.cls.cpu().numpy().astype(int)
                ppe_confidences = ppe_result.boxes.conf.cpu().numpy()    

                # Adjust PPE boxes to original image coordinates
                for i in range(len(ppe_boxes)):
                    ppe_boxes[i] = [ppe_boxes[i][0] + x_min, ppe_boxes[i][1] + y_min, ppe_boxes[i][2] + x_min, ppe_boxes[i][3] + y_min]

                draw_boxes(image, ppe_boxes, ppe_labels, ppe_confidences, ppe_label_mapping, confidence_threshold=0.6)

        draw_boxes(image, person_boxes, person_labels, person_confidences, person_label_mapping, confidence_threshold=0.7)

    # Save the output image
    cv2.imwrite(output_path, image)

def main(input_dir, output_dir, person_model_path, ppe_model_path):
    # Load models
    person_model = YOLO(person_model_path)
    ppe_model = YOLO(ppe_model_path)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process each image in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            process_image(image_path, output_path, person_model, ppe_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Inference Script")
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the input directory containing images')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save the output images')
    parser.add_argument('--person_model_path', type=str, required=True, help='Path to the person detection model')
    parser.add_argument('--ppe_model_path', type=str, required=True, help='Path to the PPE detection model')
    
    args = parser.parse_args()
    
    main(args.input_dir, args.output_dir, args.person_model_path, args.ppe_model_path)