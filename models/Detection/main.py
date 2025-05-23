from ultralytics import YOLO
import os

def main():
    data_yaml    = 'car_plate.yaml'            
    base_model   = 'yolo11n.pt'                  
    project_dir  = 'output'                      
    run_name     = 'exp1'                       
    epochs       = 50
    imgsz        = 640
    batch_size   = 16

    model = YOLO(base_model)
    model.train(
        data   = data_yaml,
        epochs = epochs,
        imgsz  = imgsz,
        batch  = batch_size,
        project= project_dir,
        name   = run_name
    )
    print(f"Training finished. Weights saved to {project_dir}/{run_name}/weights/best.pt")


if __name__ == '__main__':
    main()
