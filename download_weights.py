import os
import urllib.request

def download_yolo_weights():
    weights_url = "https://pjreddie.com/media/files/yolov3.weights"
    weights_path = "yolov3.weights"
    
    if not os.path.exists(weights_path):
        print("Downloading YOLOv3 weights...")
        urllib.request.urlretrieve(weights_url, weights_path)
        print("Download complete!")
    else:
        print("YOLOv3 weights already exist.")

if __name__ == "__main__":
    download_yolo_weights() 