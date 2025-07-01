# GPOD Authentication System

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Flask](https://img.shields.io/badge/Backend-Flask-lightgrey.svg)
![YOLOv3](https://img.shields.io/badge/Object%20Detection-YOLOv3-red.svg)
![Status: Active](https://img.shields.io/badge/Status-Active-brightgreen.svg)

Project Overview
GPOD (Graphical Password with Object Detection) is a secure, visual authentication system that replaces traditional text-based passwords with graphical challenges. The system leverages YOLOv3 for object detection, allowing users to authenticate by selecting specific objects in an image. This project provides an intuitive, modern alternative to conventional login mechanisms, enhancing both security and user experience.

Features-
Graphical password-based authentication

Real-time object detection using YOLOv3

Modern, responsive user interface with Tailwind CSS

Encrypted user data storage

Interactive click effects and visual feedback

Session timeout management for added security

Tech Stack-
Python 3.8+

Flask

PyTorch

OpenCV

YOLOv3

Tailwind CSS
![loginpage](https://github.com/user-attachments/assets/736f42ab-b307-41c2-ad5d-0cc92411e32f)
![challenge generation](https://github.com/user-attachments/assets/37a35dd0-86f8-456b-ad6d-b76e342f3780)
![successful](https://github.com/user-attachments/assets/04849fe4-827d-46ba-a817-65c5399ba74f)

Prerequisites-
Python 3.8+

Flask

PyTorch

OpenCV

Installation Steps-
Clone the repository:
git clone https://github.com/gayathrym/gpod.git
cd gpod

Install the required dependencies:
pip install -r requirements.txt

Download YOLOv3 weights:
wget https://www.kaggle.com/datasets/shivam316/yolov3-weights

Setup Tutorial
Clone the project and navigate to the directory.

Install all dependencies listed in requirements.txt.

Download and place the YOLOv3 weights in the project root or as required by the YOLO utility functions.

Start the Flask server:
python app.py
Open the application in your browser:
Register a new user by entering a username, selecting a background, and choosing object categories.
Login using the username and by clicking on the correct objects in the challenge image.
If authentication succeeds, access is granted. Otherwise, retry the challenge

Project Structure-
gpod-auth/
├── app.py              # Main Flask application
├── static/             # CSS, JavaScript, Images
│   ├── css/
│   ├── js/
│   └── images/
├── templates/          # HTML templates
│   ├── base.html
│   ├── register.html
│   └── login.html
├── yolo_utils.py       # YOLO object detection utilities
├── gpod_logic.py       # Authentication logic
├── requirements.txt    # Python package requirements
└── README.md

Security Notes
User data is securely encrypted.
Graphical password system uses challenge-response authentication.
Click-based position verification prevents automated attacks.
Session timeout is implemented to protect against unauthorized access.
