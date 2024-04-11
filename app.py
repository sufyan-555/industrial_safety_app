import cv2
import base64
from flask import Flask, render_template, Response,request,redirect
from models.r_zone import people_detection
from datetime import datetime,timedelta

from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///alerts.db'
app.config["SQLALCHEMY_BINDS"]={"complain":"sqlite:///complain.db"}

db = SQLAlchemy(app)


class Alert(db.Model):
    id = db.Column(db.Integer)
    date_time = db.Column(db.DateTime,primary_key=True)
    alert_type = db.Column(db.String(50))
    frame_snapshot = db.Column(db.LargeBinary)

## deal with unique key and personid key

class Complaint(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(100))
    email = db.Column(db.String(100))
    alert_type = db.Column(db.String(50))
    description = db.Column(db.Text)
    file_data = db.Column(db.LargeBinary)

cameras={}

r_zone=people_detection("/models/yolov8n.pt")



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

        # frame processing for restricted Zone
        results=r_zone.process(img=frame,region=region,flag=flag_r_zone)
        if results[0]:
            for person in results[1]:
                x1,y1,x2,y2=person[0]
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)

            with app.app_context():
                latest_alert = Alert.query.filter_by(alert_type='restricted_zone').order_by(Alert.date_time.desc()).first()

                if ((latest_alert is None) or ((datetime.now() - latest_alert.date_time) > timedelta(minutes=1))):
                    new_alert = Alert(date_time=datetime.now(), alert_type='restricted_zone', 
                                      frame_snapshot=cv2.imencode('.jpg', frame)[1].tobytes())
                    db.session.add(new_alert)
                    db.session.commit()

                    
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login')
def login():
    return redirect('/dashboard')

@app.route('/submit_complaint', methods=['POST'])
def submit_complaint():
    if request.method == 'POST':
        full_name = request.form['fullName']
        email = request.form['email']
        alert_type = request.form['alertType']
        description = request.form['description']
        file_data = request.files['file'].read() if 'file' in request.files else None
      

        complaint = Complaint(full_name=full_name, email=email, alert_type=alert_type, description=description, file_data=file_data)
        db.session.add(complaint)
        db.session.commit()

        return redirect('/')


@app.route('/dashboard')
def dash_page():
    return render_template('dash.html',cameras=cameras)

@app.route("/manage_camera")
def manage_cam_page():
    return render_template('manage_cam.html')

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
        "region": False
    }
        
    return redirect("/dashboard")


@app.route('/notifications')
def notifications():
    alerts =  Alert.query.order_by(Alert.date_time.desc()).all()
    for alert in alerts:
        alert.frame_snapshot = base64.b64encode(alert.frame_snapshot).decode('utf-8')
    return render_template('notifications.html', alerts=alerts)

@app.route('/complaints')
def complaints():
    complaints = Complaint.query.all()
    for complaint in complaints:
        if complaint.file_data:
            # Convert binary file data to base64 for displaying in HTML
            complaint.file_data = base64.b64encode(complaint.file_data).decode('utf-8')
    return render_template('complaints.html', complaints=complaints)



@app.route('/delete/<int:id>')
def delete(id):
    complaint=Complaint.query.filter_by(id=id).first()
    db.session.delete(complaint)
    db.session.commit()
    return redirect("/complaints")



@app.route('/delete_notification/<int:id>')
def delete_notification(id):
    alert=Alert.query.filter_by(id=id).first()
    db.session.delete(alert)
    db.session.commit()
    return redirect("/notifications")



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


@app.route('/logout')
def logout():
    return redirect('/')


if __name__ == "__main__":
    app.run(debug=True,port=8000)

