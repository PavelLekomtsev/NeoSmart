import os
import random
import xml.etree.ElementTree as ET
import cv2
import cvzone
import numpy as np

# Define accepted image formats
acceptedFormats = ['.jpg', '.png', '.jpeg', '.bmp']
totalBbox = 0

# Utility function to get the color for a given class
def getColorForClass(classId, num_classes):
    colors = cv2.applyColorMap(np.linspace(0, 255, num_classes).astype(np.uint8), cv2.COLORMAP_JET)
    return tuple(map(int, colors[classId][0]))[::-1]

# Function to draw bounding boxes on images annotated in YOLO format
def drawBoxesYOLO(imagePath, fileExtension, class_names):
    global totalBbox
    img = cv2.imread(imagePath)
    h, w, _ = img.shape
    with open(imagePath.replace(fileExtension, ".txt"), 'r') as f:
        lines = f.readlines()

        for line in lines:
            totalBbox += 1
            classId, xCenter, yCenter, width, height = map(float, line.strip().split())
            classId = int(classId)
            xmin = int((xCenter - width / 2) * w)
            ymin = int((yCenter - height / 2) * h)
            xmax = int((xCenter + width / 2) * w)
            ymax = int((yCenter + height / 2) * h)

            color = getColorForClass(classId, len(class_names))
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 4)
            cvzone.putTextRect(img, class_names[classId], (xmin + 10, ymin - 10))

    return img

# Function to draw bounding boxes on images annotated in VOC format
def drawBoxesVOC(imagePath, fileExtension, class_names):
    img = cv2.imread(imagePath)
    tree = ET.parse(imagePath.replace(fileExtension, ".xml"))
    root = tree.getroot()

    for obj in root.findall("object"):
        class_name = obj.find("name").text
        classId = class_names.index(class_name)

        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)

        color = getColorForClass(classId, len(class_names))
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 4)
        cvzone.putTextRect(img, class_name, (xmin + 10, ymin - 10))

    return img

# Function to display a subset or navigate through all the dataset images
def displayDetectionDatasetSamples(dataFolder, annotationType="YOLO", imagesToDisplay=50, scale=0.3, imagesPerCol=5, class_names=None):
    allImages = [f for f in os.listdir(dataFolder) if any(f.endswith(ext) for ext in acceptedFormats)]

    if not class_names:
        print("Warning: Class names list is empty. No class names will be displayed.")

    # Mode 1: Display a random set of annotated images
    if imagesToDisplay:
        selectedImages = random.sample(allImages, min(imagesToDisplay, len(allImages)))
        imageList = []

        for imgFile in selectedImages:
            imagePath = os.path.join(dataFolder, imgFile)
            fileExtension = os.path.splitext(imgFile)[1]

            if annotationType == "YOLO":
                img = drawBoxesYOLO(imagePath, fileExtension, class_names)
            elif annotationType == "VOC":
                img = drawBoxesVOC(imagePath, fileExtension, class_names)

            imageList.append(img)

        # Stack and display the selected images
        imgStacked = cvzone.stackImages(imageList, imagesPerCol, scale)
        cv2.imshow('Stacked Images', imgStacked)
        cv2.waitKey(0)

    # Mode 2: Navigate through each image one by one
    else:
        idx = 0
        while idx < len(allImages):
            imgFile = allImages[idx]
            imagePath = os.path.join(dataFolder, imgFile)
            fileName, fileExtension = os.path.splitext(imgFile)

            if annotationType == "YOLO":
                img = drawBoxesYOLO(imagePath, fileExtension, class_names)
            elif annotationType == "VOC":
                img = drawBoxesVOC(imagePath, fileExtension, class_names)

            print(fileName)
            cv2.imshow("Image", img)
            key = cv2.waitKey(0)
            # Navigation controls
            if key == ord('d'):  # Delete image and annotation
                os.remove(imagePath)
                if annotationType == "YOLO":
                    os.remove(imagePath.replace(fileExtension, ".txt"))
                elif annotationType == "VOC":
                    os.remove(imagePath.replace(fileExtension, ".xml"))
                del allImages[idx]
                continue
            elif key == ord('w'):  # Next image
                idx += 1
            elif key == ord('s'):  # Previous image
                idx -= 1

            if idx >= len(allImages):
                break

# Main execution
if __name__ == "__main__":
    displayDetectionDatasetSamples(dataFolder="raw",
                                   annotationType="YOLO",
                                   imagesToDisplay=None,
                                   scale=0.5, imagesPerCol=4,
                                   class_names=["car"])

