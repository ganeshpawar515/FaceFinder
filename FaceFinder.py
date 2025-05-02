import cv2
import os
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
import time

# Parameters
FACE_DISTANCE_THRESHOLD = 0.5 #to determine unique or not if < same
FRAME_SKIP = 1
FRAME_RESIZE = (640, 360)
MIN_CONFIDENCE = 0.5  # Minimum confidence for face detection


# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# MTCNN and FaceNet
mtcnn = MTCNN(keep_all=False, min_face_size=20, thresholds=[0.1, 0.1, 0.1], device=DEVICE)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

def euclidean_distance(embedding1, embedding2):
    return np.linalg.norm(embedding1 - embedding2)

def is_unique_face(face_embedding, known_embeddings, threshold):
    if not known_embeddings:
        return True
    distances = [euclidean_distance(face_embedding, known_embedding) for known_embedding in known_embeddings]
    return all(distance > threshold for distance in distances)

def process_input(video_capture, output_dir, display=False):
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT)) if video_capture.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else float('inf')
    
    frame_count = 0
    person_count = 0
    known_embeddings = []
    recent_faces = []  # To store embeddings for a short-term memory

    os.makedirs(output_dir, exist_ok=True)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        resized_frame = cv2.resize(frame, FRAME_RESIZE)
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

        boxes, probs = mtcnn.detect(rgb_frame)

        if boxes is not None:
            for i, (box, prob) in enumerate(zip(boxes, probs)):
                if prob < MIN_CONFIDENCE:
                    continue  # Skip low-confidence detections

                left, top, right, bottom = [int(val) for val in box]
                face_image = resized_frame[top:bottom, left:right]

                # Skip invalid or out-of-bounds crops
                if face_image.shape[0] == 0 or face_image.shape[1] == 0:
                    continue

                face_resized = cv2.resize(face_image, (160, 160))
                face_tensor = torch.tensor(np.transpose(face_resized, (2, 0, 1)), dtype=torch.float).unsqueeze(0).to(DEVICE)
                face_embedding = facenet(face_tensor).detach().cpu().numpy().flatten()
                face_embedding /= np.linalg.norm(face_embedding)

                # Check uniqueness against long-term and short-term memory
                if (is_unique_face(face_embedding, known_embeddings, FACE_DISTANCE_THRESHOLD) and
                        is_unique_face(face_embedding, recent_faces, FACE_DISTANCE_THRESHOLD)):
                    known_embeddings.append(face_embedding)
                    recent_faces.append(face_embedding)
                  
                    person_count += 1
                    face_path = os.path.join(output_dir, f"person_{person_count}.jpg")
                    cv2.imwrite(face_path, face_resized)

                # Draw bounding box on the frame
                cv2.rectangle(resized_frame, (left, top), (right, bottom), (0, 255, 0), 2)

        if display:
            # Show the frame with bounding boxes
            cv2.imshow("Live Feed", resized_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        elapsed_time = time.time() - time.time()
        percentage = (frame_count / total_frames) * 100 if total_frames != float('inf') else 0
        print(f"Processed {percentage:.2f}% - Elapsed: {elapsed_time:.2f}s", end='\r')

    video_capture.release()
    if display:
        cv2.destroyAllWindows()
    print(f"\nTotal unique faces saved: {person_count}")

if __name__ == "__main__":
    choice = input("Do you want to use live webcam feed? (yes/no): ").strip().lower()

    if choice == 'yes':
        print("Starting live feed...")
        video_capture = cv2.VideoCapture(0)  # Webcam feed
        output_dir = "live_feed_extracted"
        display = True
    else:
        video_path = input("Enter the path to the video file: ").strip()
        if not os.path.exists(video_path):
            print(f"Error: The file '{video_path}' does not exist.")
            exit()
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = f"{video_name}_extracted"
        video_capture = cv2.VideoCapture(video_path)
        display = False

    process_input(video_capture, output_dir, display=display)
