import os
import cv2
import base64
from flask import Flask, render_template, Response, request, redirect, flash, session
from datetime import datetime, timedelta
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user

from models.r_zone import people_detection
from models.fire_detection import fire_detection
from models.gear_detection import gear_detection
from models.alert import alert
from models.motion_amp import amp

app = Flask(__name__)
app.config['SECRET_KEY'] = 'the random string'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///user.db'
app.config["SQLALCHEMY_BINDS"]={"complain":"sqlite:///complain.db",
                                "cams":"sqlite:///cams.db",
                                "alerts":"sqlite:///alerts.db"}


app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {"mp4"}

db = SQLAlchemy(app)
login_manager = LoginManager(app)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(UserMixin, db.Model):  # User table
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))
    email = db.Column(db.String(100))

class Camera(db.Model):  # Camera table
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))  # Foreign key referencing User table
    cam_id = db.Column(db.String(100))
    fire_detection = db.Column(db.Boolean, default=False)
    pose_alert = db.Column(db.Boolean, default=False)
    restricted_zone = db.Column(db.Boolean, default=False)
    safety_gear_detection = db.Column(db.Boolean, default=False)
    region = db.Column(db.Boolean, default=False)

class Alert(db.Model):  # Alert table
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))  # Foreign key referencing User table
    date_time = db.Column(db.DateTime)
    alert_type = db.Column(db.String(50))
    frame_snapshot = db.Column(db.LargeBinary)

class Complaint(db.Model):  # Complaint table
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))  # Foreign key referencing User table
    full_name = db.Column(db.String(100))
    email = db.Column(db.String(100))
    alert_type = db.Column(db.String(50))
    description = db.Column(db.Text)
    file_data = db.Column(db.LargeBinary)

r_zone=people_detection("models/yolov8n.pt")
fire_det=fire_detection("models/fire.pt",conf=0.60)
gear_det=gear_detection("models/gear.pt")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login_page')
def login_page():
    return render_template("login.html")

@app.route('/register_page')
def register_page():
    return render_template("register.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        if(not email or not password):
            flash('Email or Password Missing!!')
            return redirect('/login')
        
        user = User.query.filter_by(email=email).first()
        
        if user and user.password == password:
            login_user(user)
            return redirect('/dashboard')
        else:
            flash('Invalid email or password')
            return redirect('/login')

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['name']
        email = request.form['email']
        password = request.form['password']
        
        # Check if the username or email already exists in the database
        existing_user = User.query.filter((User.username == username) | (User.email == email)).first()
        if existing_user:
            flash('Username or email already exists.')
            return redirect('/register')

        # Create a new user
        new_user = User(username=username, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        
        flash('User registered successfully! Please log in.')
        return redirect('/login')

    return render_template('register.html')

@app.route('/upload')           #routing for video upload
def upload():
    return render_template('VideoUpload.html')

@app.route('/upload_file', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect("/upload")
        file = request.files['file']
        if file.filename == '':
            flash('No File Selected')
            return redirect("/upload")
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            in_path=f"uploads/{filename}"
            out_path=f"static/outs/output.avi"
            os.remove(out_path)
            amp(in_path=in_path,out_path=out_path,alpha=2.5,beta=0.5,m=3)
            os.remove(in_path)
            
            flash(f"Your processed video is available <a href='/{out_path}' target='_blank'>here</a>")
            return redirect("/upload")
        else:
            flash("File in wrong Format!!")
            return redirect("/upload")


@app.route('/<int:id>/submit_complaint', methods=['POST','GET'])
def submit_complaint(id):
    if request.method == 'POST':
        full_name = request.form['fullName']
        email = request.form['email']
        alert_type = request.form['alertType']
        description = request.form['description']
        file_data = request.files['file'].read() if 'file' in request.files else None

        # Create a new complaint associated with the logged-in user
        complaint = Complaint(full_name=full_name, email=email, alert_type=alert_type, description=description,
                              file_data=file_data, user_id=id)
        db.session.add(complaint)
        db.session.commit()
        link=f'/complain/{id}'
        flash("Your Complaint has been recorded, We'll get back to you soon.")
        return redirect(link)

@app.route('/dashboard')
@login_required
def dash_page():
    # Query cameras belonging to the logged-in user
    cameras = Camera.query.filter_by(user_id=current_user.id).all()
    return render_template('dash.html', cameras=cameras)

@app.route("/manage_camera")
@login_required
def manage_cam_page():
    # Query cameras belonging to the logged-in user
    cameras = Camera.query.filter_by(user_id=current_user.id).all()
    return render_template('manage_cam.html', cameras=cameras)

@app.route("/get_cam_details", methods=['GET', 'POST'])
@login_required
def getting_cam_details():
    if request.method == 'POST':
        camid = request.form['Cam_id']

        fire_bool = "fire" in request.form
        pose_bool = "pose_alert" in request.form
        r_bool = "R_zone" in request.form
        s_gear_bool = "Safety_gear" in request.form

        # Check if the camera details already exist in the database for the logged-in user
        camera = Camera.query.filter_by(cam_id=camid, user_id=current_user.id).first()

        if camera:
            # Update existing camera details
            camera.fire_detection = fire_bool
            camera.pose_alert = pose_bool
            camera.restricted_zone = r_bool
            camera.safety_gear_detection = s_gear_bool
        else:
            # Create a new camera entry associated with the logged-in user
            camera = Camera(user_id=current_user.id, cam_id=camid, fire_detection=fire_bool, pose_alert=pose_bool,
                            restricted_zone=r_bool, safety_gear_detection=s_gear_bool)

        # Commit changes to the database
        db.session.add(camera)
        db.session.commit()
    return redirect("/manage_camera")

@app.route('/notifications')
@login_required
def notifications():
    # Query notifications belonging to the logged-in user
    alerts = Alert.query.filter_by(user_id=current_user.id).order_by(Alert.date_time.desc()).all()
    for alert in alerts:
        alert.frame_snapshot = base64.b64encode(alert.frame_snapshot).decode('utf-8')
    return render_template('notifications.html', alerts=alerts)

@app.route('/complaints')
@login_required
def complaints():
    # Query complaints belonging to the logged-in user
    complaints = Complaint.query.filter_by(user_id=current_user.id).all()
    for complaint in complaints:
        if complaint.file_data:
            # Convert binary file data to base64 for displaying in HTML
            complaint.file_data = base64.b64encode(complaint.file_data).decode('utf-8')
    return render_template('complaints.html', complaints=complaints,user=current_user)

@app.route("/complain/<int:id>")
def complain_form(id):
    user = User.query.filter_by(id=id).first()
    return render_template("complain_form.html",username=user.username,id=user.id)


@app.route('/delete/<int:id>')                  
@login_required
def delete(id):
    # Delete complaint belonging to the logged-in user
    complaint = Complaint.query.filter_by(id=id, user_id=current_user.id).first()
    db.session.delete(complaint)
    db.session.commit()
    return redirect("/complaints")

@app.route('/delete_notification/<int:id>')         
@login_required
def delete_notification(id):
    # Delete notification belonging to the logged-in user
    alert = Alert.query.filter_by(id=id, user_id=current_user.id).first()
    db.session.delete(alert)
    db.session.commit()
    return redirect("/notifications")

@app.route('/delete_camera/<int:id>')               
@login_required
def delete_camera(id):
    # Delete camera belonging to the logged-in user
    camera = Camera.query.filter_by(id=id, user_id=current_user.id).first()
    db.session.delete(camera)
    db.session.commit()
    return redirect("/manage_camera")

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect('/')

@app.route('/video_feed/<string:cam_id>')
@login_required
def video_feed(cam_id):
    camera = Camera.query.filter_by(cam_id=str(cam_id), user_id=current_user.id).first()
    if camera:
        flag_r_zone = camera.restricted_zone
        flag_pose_alert = camera.pose_alert
        flag_fire = camera.fire_detection
        flag_gear = camera.safety_gear_detection
        region = camera.region
        
        try:
            return Response(process_frames(str(cam_id), region, flag_r_zone, flag_pose_alert,
                                       flag_fire, flag_gear, current_user.id), mimetype='multipart/x-mixed-replace; boundary=frame')
        except:
            return "Something wrong with Cam Details !!"
    else:
        return "Camera details not found."

def add_to_db(results, frame, alert_name, user_id=None):
    if results[0]:
        for box in results[1]:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        with app.app_context():
            latest_alert = Alert.query.filter_by(alert_type=alert_name, user_id=user_id).order_by(Alert.date_time.desc()).first()

            if (latest_alert is None) or ((datetime.now() - latest_alert.date_time) > timedelta(minutes=1)):
                new_alert = Alert(date_time=datetime.now(), alert_type=alert_name, frame_snapshot=cv2.imencode('.jpg', frame)[1].tobytes(), user_id=user_id)
                db.session.add(new_alert)
                db.session.commit()


def process_frames(camid, region, flag_r_zone=False, flag_pose_alert=False, flag_fire=False, flag_gear=False, user_id=None):
    if len(camid) == 1:
        camid = int(camid)
        cap = cv2.VideoCapture(camid)
        ret = True
    else:
        address = f"http://{camid}/video"
        print(address)
        cap = cv2.VideoCapture(0)
        cap.open(address)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1000, 500))

        # frame processing for restricted Zone
        results = r_zone.process(img=frame, region=region, flag=flag_r_zone)
        add_to_db(results=results, frame=frame, alert_name="restricted_zone", user_id=user_id)

        # fire detection
        results = fire_det.process(img=frame, flag=flag_fire)
        add_to_db(results=results, frame=frame, alert_name="fire_detection", user_id=user_id)

        # gear detection
        results = gear_det.process(img=frame, flag=flag_gear)
        add_to_db(results=results, frame=frame, alert_name="gear_detection", user_id=user_id)

        #alert
        results=alert(frame=frame,flag=flag_pose_alert)
        add_to_db(results=results,frame=frame,alert_name="Pose Alert",user_id=user_id)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


if __name__=="__main__":
    app.run(debug=True)

