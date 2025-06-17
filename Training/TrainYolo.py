from ultralytics import YOLO

def main():
    model = YOLO('yolov8s.pt')
    model.train(data='data/data.yaml',
                epochs=100,
                imgsz=640,
                )

if __name__ == '__main__':
    main()
