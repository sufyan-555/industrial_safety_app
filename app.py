import cv2
from flask import Flask, render_template, Response,request,redirect

app = Flask(__name__)

cameras={"0":{
    "region":[],
    "flag_r_zone":True,
    "flag_pose_alert":False,
    "flag_fire":False,
    "flag_gear":False
}}

def process_frames(camid,region,flag_r_zone=False,flag_pose_alert=False,flag_fire=False,flag_gear=False):
    """
    function to process frames

    Args:
    camid: camid given by the user 
    flag_people: Flag for peolpe_detection 
    flag_vehicle: Flag for vehicle detection
    flag_fiere: Flag for fire detection 
    flag_smoke: Flag for smoke detection

    returns: image object

    """
    if  (len(camid)==1):
        camid=int(camid)
    
    cap=cv2.VideoCapture(camid)
    ret=True
    while(True):
        ret,frame=cap.read()
        if not ret:
            break
        
        frame=cv2.resize(frame,(600,300))
        
        frame=cv2.flip(frame,1)


        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')





@app.route('/')
def dash():
    return render_template('dash.html',cameras=cameras)





@app.route('/video_feed/<int:cam_id>')
def video_feed(cam_id):
    camid=str(cam_id)
    flag_r_zone=cameras[camid]["flag_r_zone"]
    flag_pose_alert=cameras[camid]["flag_pose_alert"]
    flag_fire=cameras[camid]["flag_fire"]
    flag_gear=cameras[camid]["flag_gear"]
    region=cameras[camid]["region"]

    return Response(process_frames(camid,region,flag_r_zone,flag_pose_alert,
                                   flag_fire,flag_gear), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)



""""
cam id: str
fire: check button
pose alert: check button
restricted zone: check button
safety gear: check button

"""