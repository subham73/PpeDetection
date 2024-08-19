import os
import xml.etree.ElementTree as ET
import argparse

def convert_voc_to_yolo(input_dir, output_dir, image_id, classes, requested_classes):

    annotation_path = open(f'{input_dir}/{image_id}.xml')
    # out_file = open(f'{output_dir}/{image_id}.txt', 'w')
    out_file = open(f'{output_dir}/{image_id}.txt', 'w')

    # Parse the XML annotation file
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    # Get image dimensions
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    yolo_annotations = []

    for obj in root.findall('object'):
        # Get the class label
        class_name = obj.find('name').text
        class_id = classes.index(class_name)

        # Skip this object if it is not in the requested classes
        # if class_name not in requested_classes:
        #     continue

        # Get the bounding box coordinates
        bndbox = obj.find('bndbox')
        x_min = int(bndbox.find('xmin').text)
        y_min = int(bndbox.find('ymin').text)
        x_max = int(bndbox.find('xmax').text)
        y_max = int(bndbox.find('ymax').text)

        # Convert to YOLO format
        x_center = (x_min + x_max) / 2.0 / width
        y_center = (y_min + y_max) / 2.0 / height
        bbox_width = (x_max - x_min) / float(width)
        bbox_height = (y_max - y_min) / float(height)

        yolo_annotations.append(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}")

    out_file.write("\n".join(yolo_annotations))



if __name__ == '__main__':
    # taking input and output directories as arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', help='Path to the input directory')
    parser.add_argument('output_dir', help='Path to the output directory')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # all the classes in the annotion files
    classes = [line.strip() for line in open(f'datasets/classes.txt')]

    # for preparing specific annotation files
    requested_classes = []

    for image_id in os.listdir(f'{args.input_dir}'):
        image_id = image_id.replace('.xml', '')

        if len(requested_classes) == 0:
            convert_voc_to_yolo(args.input_dir, args.output_dir, image_id, classes, classes)
        else:
            convert_voc_to_yolo(args.input_dir, args.output_dir, image_id, classes, requested_classes)

    total_images = len(os.listdir(f'{args.output_dir}'))

    print(f'Converted.. {total_images} images' )


