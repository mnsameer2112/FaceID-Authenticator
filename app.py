from flask import Flask,request,redirect,url_for,render_template
from werkzeug.utils import secure_filename
from mtcnn import MTCNN
import cv2
import base64
from io import BytesIO
import numpy as np
from keras_facenet import FaceNet
import os
from PIL import Image
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app=Flask(__name__)

def Face_id(image_path1,image_path2):
    detector=MTCNN()
    facenet=FaceNet()
    image_paths=[image_path1,image_path2]
    dets=[]
    detections=[]
    for img in image_paths:
        img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
        detections.append(detector.detect_faces(img))
    dets=[detections[0][0]['box'],detections[1][0]['box']]
    faces=[]
    x1,y1,w1,h1=dets[0]
    original_image1=cv2.imread(image_path1)
    image_rgb1 = cv2.cvtColor(original_image1, cv2.COLOR_BGR2RGB)
    face1 = image_rgb1[y1:y1+h1, x1:x1+w1]
    faces.append(face1)
    x2,y2,w2,h2=dets[1]
    original_image2=cv2.imread(image_path2)
    image_rgb2 = cv2.cvtColor(original_image2, cv2.COLOR_BGR2RGB)
    face2 = image_rgb2[y2:y2+h2, x2:x2+w2]
    faces.append(face2)
    embs=[]
    for face in faces:
        face=cv2.resize(face,(160,160))
        face=np.expand_dims(face,axis=0)
        embeddings=facenet.embeddings(face)
        embs.append(embeddings[0])
    distance=facenet.compute_distance(embs[0],embs[1])
    return distance
       
        
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/fid', methods=['POST'])
def fid():
    f1 = request.files['image']

    filename1 = secure_filename(f1.filename)
    save_path1 = os.path.join('C:\\Users\\HP\\FaceID_app', filename1)
    f1.save(save_path1)

    # Handle captured image data correctly (either from files or base64)
    if 'captured_image_data' in request.files:
        captured_image_data = request.files['captured_image_data']
        image_data = captured_image_data.read() 
    else:
        captured_image_data = request.form['captured_image_data'].split(',')[1]
        image_data = base64.b64decode(captured_image_data)
    captured_image = Image.open(BytesIO(image_data))
    captured_image_path = os.path.join('C:\\Users\\HP\\FaceID_app', 'captured_image.jpg')
    captured_image.save(captured_image_path , 'JPEG')

    distance = Face_id(save_path1, captured_image_path)
    similarity = (1-distance)*100
    if distance<0.45:
        return f"Person verified successfully,Similarity: {similarity:.2f}"
    else:
        return f"The faces are not the same. Similarity: {similarity:.2f}%"


if __name__=='main':
    app.run(debug=True)