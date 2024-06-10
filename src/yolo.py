from ultralytics import YOLO
import cv2
from PIL import Image

# Load a model
model = YOLO("yolov8m.yaml")  # build a new model from scratch
model = YOLO("yolov8m.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="data_3.yaml", epochs=1000, batch=64, patience=200)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
path = model.export(format="onnx")  # export the model to ONNX format

# Retrieve and save the results
results[0].save("output_image.jpg")

# Display the image with bounding boxes
output_img = cv2.imread("output_image.jpg")
cv2.imshow("YOLO Prediction", output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()