import numpy as np
import cv2

def process_frame(frame, buffer, alpha, beta):
    avg_intensity = np.mean(buffer, axis=0)
    diff = frame - avg_intensity
    output_frame = beta * frame + alpha * diff
    output_frame = np.clip(output_frame, 0, 255)
    return output_frame.astype(np.uint8)

def amp(in_path,out_path,alpha=2.5,beta=0.7,m=5):
    cap = cv2.VideoCapture(in_path)       
    buffer = []

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = frame.astype(np.float32)
        
        if len(buffer) < m:
            buffer.append(frame)
            continue
        
        if len(buffer) == m:
            processed_frame = process_frame(frame, np.array(buffer), alpha, beta)
            out.write(processed_frame)
        
        buffer.pop(0)
        buffer.append(frame)
    cap.release()
    out.release()
    return(True)
