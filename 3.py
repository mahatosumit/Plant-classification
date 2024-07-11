import os
import yolov8

def train_yolov8(data_path, cfg_path, weights_path=None, batch_size=16, epochs=50):
    # Set the path to the dataset configuration file
    data = {
        'train': os.path.join(r'G:\SUMIT\experment\modelv1\test'),
        'val': os.path.join(r'G:\SUMIT\experment\modelv1\valid'),
        'nc': 2,  # Number of classes
        'names': ['class1', 'class2']  # Class names
    }

    # Training configuration
    train_cfg = {
        'batch_size': batch_size,
        'epochs': epochs,
        'data': data,
        'cfg': cfg_path,  # Model configuration file (e.g., yolov8n.yaml)
        'weights': weights_path,  # Pre-trained weights, can be None to start from scratch
        'device': 'cuda',  # Or 'cpu'
    }

    # Initialize model
    model = yolov8.YOLOv8(train_cfg)

    # Start training
    model.train()

if __name__ == '__main__':
    # Define paths
    DATA_PATH = r"G:\SUMIT\experment\modelv1"
    CFG_PATH = r'G:\SUMIT\experment\data.yaml'  # Path to the YOLOv8 config file
    WEIGHTS_PATH = None  # Path to the pre-trained weights or None

    # Start training process
    train_yolov8(DATA_PATH, CFG_PATH, WEIGHTS_PATH)
