from ultralytics import YOLO

#model = YOLO()
# If you want to finetune the model with pretrained weights, you could load the 
# pretrained weights like below
# model = YOLOv10.from_pretrained('jameslahm/yolov10{n/s/m/b/l/x}')
# or
# wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10{n/s/m/b/l/x}.pt
# model = YOLOv10('yolov10{n/s/m/b/l/x}.pt')
# Choose which model you want to use. remember larger the model higher the computational cost
#model = YOLOv10(yolo10n.pt)

#https://medium.com/@sudiplaudari/yolov10-a-step-by-step-guide-to-object-detection-on-a-custom-dataset-9f3e3e56921c

def main():
    model = YOLO('yolov10x.pt')
    print(f"Training start...")
    model.train(data='dataset.yaml', epochs=10, batch=8, imgsz=640,freeze=10,pretrained=True)
    print(f"Training done!")

if __name__ == '__main__':
    main()

