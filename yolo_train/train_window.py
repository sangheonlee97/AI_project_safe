from ultralytics import YOLO

def main():
    # Load a model
    model = YOLO("yolov8n.yaml")  # build a new model from scratch
    model = YOLO("yolov8s.pt")  # load a pretrained model (recommended for training)

    # Use the model
    model.train(data="label8.yaml", epochs=500, batch=100, patience=100)  # train the model
    metrics = model.val()  # evaluate model performance on the validation set
    results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    path = model.export(format="onnx")  # export the model to ONNX format

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
