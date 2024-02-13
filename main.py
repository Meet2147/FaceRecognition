from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np
import face_recognition
from typing import Dict

app = FastAPI()

# Stores encodings and their corresponding names
face_encodings_db: Dict[str, np.ndarray] = {}

@app.post("/upload_face/")
async def upload_face(name: str, file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    try:
        encode = face_recognition.face_encodings(img)[0]
        face_encodings_db[name] = encode
        return {"message": f"Face encoding for {name} added successfully."}
    except IndexError as e:
        raise HTTPException(status_code=400, detail="No face found in the image.")

@app.post("/recognize/")
async def recognize_face(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(img)
    encodeCurFrame = face_recognition.face_encodings(img, facesCurFrame)
    facesRecognized = []

    for encodeFace in encodeCurFrame:
        distances = face_recognition.face_distance(list(face_encodings_db.values()), encodeFace)
        if len(distances) == 0:
            continue
        best_match_index = np.argmin(distances)
        if distances[best_match_index] <= 0.6:  # Assuming 0.6 as a threshold for face matching
            best_match_name = list(face_encodings_db.keys())[best_match_index]
            facesRecognized.append(best_match_name)

    if facesRecognized:
        return JSONResponse(content={"recognized_faces": facesRecognized}, status_code=200)
    else:
        return JSONResponse(content={"message": "No recognized faces"}, status_code=200)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
