import cv2
from flask import Flask, render_template, Response,request,redirect

app = Flask(__name__)

cameras={}

def process_frames(camid,region,flag_r_zone=False,flag_pose_alert=False,flag_fire=False,flag_gear=False):
    if  (len(camid)==1):
        camid=int(camid)
    
    cap=cv2.VideoCapture(camid)
    ret=True
    while(True):
        ret,frame=cap.read()
        if not ret:
            break
        
        frame=cv2.resize(frame,(800,400))
        
        frame=cv2.flip(frame,1)


        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')





@app.route('/')
def dash_page():
    return render_template('dash.html',cameras=cameras)

@app.route("/manage_camera")
def manage_cam_page():
    return render_template('manage_cam.html')


@app.route('/video_feed/<int:cam_id>')
def video_feed_generator(cam_id):
    camid=str(cam_id)
    flag_r_zone=cameras[camid]["r_region"]
    flag_pose_alert=cameras[camid]["pose"]
    flag_fire=cameras[camid]["fire"]
    flag_gear=cameras[camid]["gear"]
    region=cameras[camid]["region"]

    return Response(process_frames(camid,region,flag_r_zone,flag_pose_alert,
                                   flag_fire,flag_gear), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/get_cam_details",methods=['GET','POST'])
def getting_cam_details():
    if(request.method=='POST'):
        camid=request.form['Cam_id']
        
        fire_bool= "fire" in request.form
        pose_bool= "pose_alert" in request.form
        r_bool= "R_zone" in request.form
        s_gear_bool= "Safety_gear" in request.form

    cameras[camid]={
        "fire": fire_bool,
        "pose": pose_bool,
        "r_region": r_bool,
        "gear": s_gear_bool,
        "region": []
    }
        
    return redirect("/")


if __name__ == "__main__":
    app.run(debug=True)

