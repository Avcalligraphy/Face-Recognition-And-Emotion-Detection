# Face Recognition and Emotion Detection

A real-time face recognition and facial emotion detection system. It detects
faces in a video stream or image, identifies known people against a registered
database, and classifies the detected emotion.

## What it does

- **Face detection and recognition** — detects faces and matches them against
  people registered in the database.
- **Emotion classification** — classifies facial expression into discrete
  emotion categories from the detected face crop.
- **Live webcam mode** — `webcam_face_recognition.py` runs the pipeline against
  a camera feed.
- **REST inference API** — a Flask backend (`app.py`) exposes the model over
  HTTP so other services can POST an image and get results back.
- **Result persistence** — `database.py` stores detection history alongside
  registered identities.

## Model

Trained with **YOLO11** for the detection stage; weights are versioned in the
repository as `best.pt`. The training and evaluation notebook lives in
[`facial-emotion-detection/`](facial-emotion-detection/), together with sample
inputs and annotated outputs under `hasil_deteksi_emosi/`.

## Stack

Python · YOLO11 · OpenCV · Flask · Jupyter

## Layout

```
app.py                            Flask inference API
webcam_face_recognition.py        live camera pipeline
database.py                       identities and detection history
best.pt                           trained model weights
facial-emotion-detection/         training notebook and evaluation samples
index.html                        minimal web client
```

## Running

```bash
pip install -r requirements.txt   # or install ultralytics, opencv-python, flask
python app.py                     # serve the inference API
python webcam_face_recognition.py # run against the local camera
```

## Notes

Detection and classification are deliberately separate stages: the detector
crops a face, the classifier scores it. That keeps the emotion model independent
of the detector, so retraining one does not invalidate the other. On CPU the
end-to-end pipeline runs comfortably in real time at modest frame rates; the
notebook documents accuracy per emotion class, where some classes are visibly
harder than others because facial expression is genuinely ambiguous, not because
the model is failing.
