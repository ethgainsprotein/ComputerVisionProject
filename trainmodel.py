from ultralytics import YOLO

def main():
    model = YOLO('yolov10x.pt')
    print(f"Training start...")
    model.train(data='dataset.yaml', epochs=10, batch=8, imgsz=640,freeze=10,pretrained=True)
    print(f"Training done!")

if __name__ == '__main__':
    main()

