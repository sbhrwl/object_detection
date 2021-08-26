## Custom Functions
* [Crop](#crop)
* [OCR](#ocr)
* [Count Objects of a class](#count-objects-of-a-class)
* [License Plate](#license-plate)
* [Commands](#commands)

### Crop
* We already have information of **Boundary box** for the detection
* Crop the image using the coordinates of the Boundary Box
```
# get box coordinates
xmin, ymin, xmax, ymax = boxes[i]
# crop detection from image (take an additional 5 pixels around all edges)
cropped_img = img[int(ymin) - 5:int(ymax) + 5, int(xmin) - 5:int(xmax) + 5]
```
* Save Cropped image
```
img_name = class_name + '_' + str(counts[class_name]) + '.png'
img_path = os.path.join(path, img_name)
cv2.imwrite(img_path, cropped_img)
```
### OCR
* Use same method as above for cropping an image
```
# get box coordinates
xmin, ymin, xmax, ymax = boxes[i]
# get the subimage that makes up the bounded region and take an additional 5 pixels on each side
box = img[int(ymin) - 5:int(ymax) + 5, int(xmin) - 5:int(xmax) + 5]
```
* Greyscale the Region within Bounding box
```
# grayscale region within bounding box
gray = cv2.cvtColor(box, cv2.COLOR_RGB2GRAY)
```
* Threshold threshold the image using **OTSUS** method to **preprocess for tesseract**
```
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
```
* Perform a median blur to smooth image slightly
```
blur = cv2.medianBlur(thresh, 3)
# resize image to double the original size as tesseract does better with certain text size
blur = cv2.resize(blur, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
```
* Run **Tesseract**
```
try:
    text = pytesseract.image_to_string(blur, config='--psm 11 --oem 3')
    print("Class: {}, Text Extracted: {}".format(class_name, text))
except:
    text = None
```
* We have Output **text**
#### Disclaimer: 
* In order to run tesseract OCR you must first download the binary files and set them up on your local machine. 
* Please do so before proceeding or commands will not run as expected!
* Official Tesseract OCR Github Repo: tesseract-ocr/tessdoc
* For Windows [Windows Install](https://github.com/UB-Mannheim/tesseract/wiki)

### Count Objects of a class
Count can work in 2 modes
* Count Total number of objects detected
* Count Total number of objects detected for a **Particular Class**
```
 boxes, scores, classes, num_objects = data
 # create dictionary to hold count of objects
 counts = dict()

 # if by_class = True then count objects per class
 if by_class:
     class_names = read_class_names(cfg.YOLO.CLASSES)

     # loop through total number of objects found
     for i in range(num_objects):
         # grab class index and convert into corresponding class name
         class_index = int(classes[i])
         class_name = class_names[class_index]
         if class_name in allowed_classes:
             counts[class_name] = counts.get(class_name, 0) + 1
         else:
             continue

 # else count total objects found
 else:
     counts['total object'] = num_objects

 return counts
```
### License Plate
[License Plate](https://github.com/sbhrwl/YoloV4_Detect_Social_Distance_Violations/blob/main/src/detection_tensorflow_framework/core/utils.py)
```
if read_plate:
    height_ratio = int(image_h / 25)
    plate_number = recognize_plate(image, coor)
    if plate_number != None:
        cv2.putText(image, plate_number, (int(coor[0]), int(coor[1] - height_ratio)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 255, 0), 2)
```

## Commands

```
# custom yolov4
python save_model.py --weights ./data/custom.weights --output ./checkpoints/custom-416 --input_size 416 --model yolov4 

# Run custom yolov4 tensorflow model
python detect.py --weights ./checkpoints/custom-416 --size 416 --model yolov4 --images ./data/images/car.jpg
python detect.py --weights ./checkpoints/yolov4-416 --size 416 --model yolov4 --images ./data/images/dog.jpg --info

# Run yolov4 on webcam
python detect_video.py --weights ./checkpoints/yolov4-416 --size 416 --model yolov4 --video 0 --output ./detections/results.avi

# Crop
python detect.py --weights ./checkpoints/yolov4-416 --size 416 --model yolov4 --images ./data/images/dog.jpg --crop

# OCR for any Image
python detect.py --weights ./checkpoints/yolov4-416 --size 416 --model yolov4 --images ./data/images/dog.jpg --ocr

# Run yolov4 model while counting total objects detected
python detect.py --weights ./checkpoints/yolov4-416 --size 416 --model yolov4 --images ./data/images/dog.jpg --count
# Run yolov4 model while counting objects per class by Upodating by_class parameter as true in count_objects method
python detect.py --weights ./checkpoints/yolov4-416 --size 416 --model yolov4 --images ./data/images/dog.jpg --count

# Run License Plate Recognition
python detect.py --weights ./checkpoints/custom-416 --size 416 --model yolov4 --images ./data/images/car2.jpg --plate
python detect_video.py --weights ./checkpoints/custom-416 --size 416 --model yolov4 --video ./data/video/license_plate.mp4 --output ./detections/recognition.avi --plate
```
## References
* [OpenCV](https://towardsdatascience.com/yolo-object-detection-with-opencv-and-python-21e50ac599e9 "OpenCV")
* [Custom Functions](https://github.com/theAIGuysCode/yolov4-custom-functions)
