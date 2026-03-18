# ProjectRoadScan
Overview

This project is a machine learning-based system designed to detect and classify road damages such as potholes and cracks using a trained YOLO model. It integrates a FastAPI backend with map visualization, chatbot interaction, and automated report generation for road condition analysis.

Features

Detects multiple types of road damage using a YOLO model

Provides severity classification based on damage type and size

Displays detected damage locations on a map using GPS coordinates

Includes a chatbot interface for querying damage insights

Generates automated road inspection reports

Stores scan history and detection data using SQLite

Technologies Used

Python

FastAPI

Uvicorn

OpenCV

NumPy

Ultralytics YOLO

SQLite

HTML / JavaScript (Frontend)

REST API

Project Structure
ProjectRoadScan/
│
├── main.py             # FastAPI backend
├── best.pt             # Trained YOLO model
├── index.html          # Frontend UI
├── road_damage.db      # SQLite database
├── scan_images/        # Stored scan images
├── requirements.txt    # Dependencies
├── .env                # Environment variables (API keys)
├── .gitignore
Installation

Clone the repository:

git clone https://github.com/your-username/ProjectRoadScan.git
cd ProjectRoadScan

Install dependencies:

pip install -r requirements.txt

Add environment variables:
Create a .env file in the root directory:

GROQ_API_KEY=your_api_key_here
ALLOWED_ORIGINS=http://localhost:8000
Running the Project

Start the FastAPI server:

uvicorn main:app --reload

Open in browser:

http://127.0.0.1:8000
API Endpoints

POST /detect
Upload an image with optional latitude, longitude, and location name to detect road damage.

GET /map-data
Retrieve all detected damage points for map visualization.

POST /chat
Interact with the chatbot using detected damage data.

POST /report
Generate a structured road damage inspection report.

GET /scans
Retrieve scan history.

GET /scans/{scan_id}
Get details of a specific scan.

DELETE /scans/{scan_id}
Delete a scan and its associated data.

GET /health
Check system status.

Model File

Ensure that the model file best.pt is present in the same directory as main.py.

Usage

Upload an image of a road surface

Provide GPS coordinates (latitude and longitude)

View detected damages with severity levels

Visualize damage points on a map

Interact with chatbot for insights

Generate a detailed inspection report

Output Classes

Alligator

Edge Cracking

Lateral Crack

Longitudinal Crack

Ravelling

Rutting

Striping

Pothole

Notes

The chatbot works with or without the Groq API (fallback logic included)

Detection accuracy depends on model training quality

Ensure correct file paths for model loading
