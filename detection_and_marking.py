import cv2
import numpy as np
import torch
from collections import deque
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from lstm import ActionClassificationLSTM

# --- SPEEDUP PARAMETERS ---
FRAME_SKIP = 1         # Process every Nth frame (0 = process all, 1 = every 2nd, etc.)
RESIZE_FACTOR = 0.5    # Resize input frames for pose detection (0.5 = half size)
USE_IMSHOW = False     # Set to True if you want to see the video live

# Load pose detection model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
pose_detector = DefaultPredictor(cfg)

SEQUENCE_LENGTH = 30
WHITE_COLOR = (255, 255, 255)
GREEN_COLOR = (0, 255, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX

CLASS_LABELS = [
    "JUMPING",
    "JUMPING_JACKS",
    "BOXING",
    "WAVING_2HANDS",
    "WAVING_1HAND",
    "CLAPPING_HANDS"
]

input_features = 36  # This matches your trained model!
hidden_dim = 128
num_classes = len(CLASS_LABELS)

model = ActionClassificationLSTM.load_from_checkpoint(
    "final_model.ckpt",
    input_features=input_features,
    hidden_dim=hidden_dim
)
model.eval().to(cfg.MODEL.DEVICE)

KEYPOINT_INDICES = [
    1,  # Left Eye
    2,  # Right Eye
    5,  # Left Shoulder
    6,  # Right Shoulder
    11, # Left Hip
    12, # Right Hip
    9,  # Left Wrist
    10, # Right Wrist
    13, # Left Knee 
    14, # Right Knee
    15, # Left Ankle
    16  # Right Ankle
]

def draw_line(image, p1, p2, color):
    cv2.line(image, tuple(map(int, p1)), tuple(map(int, p2)), color, thickness=2, lineType=cv2.LINE_AA)

def draw_keypoints(person, img):
    keypoints = person[:, :2]
    confidence = person[:, 2]
    skeleton = [
        (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 6), (5, 11), (6, 12),
        (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 12)
    ]
    for p1, p2 in skeleton:
        if confidence[p1] > 0.5 and confidence[p2] > 0.5:
            draw_line(img, keypoints[p1], keypoints[p2], GREEN_COLOR)
    for i, (x, y) in enumerate(keypoints):
        if confidence[i] > 0.5:
            cv2.circle(img, (int(x), int(y)), 4, WHITE_COLOR, -1)

def draw_action_label(frame, label):
    cv2.putText(frame, f"Action: {label}", (20, 40), FONT, 1, (0, 255, 255), 2, cv2.LINE_AA)

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fpr is None:
        fps = 15
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not out.isOpened():
        print(f"Error: Could not open {outpsut_path} for writing.")
        cap.release()
        return

    sequence = deque(maxlen=SEQUENCE_LENGTH)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if FRAME_SKIP > 0 and (frame_idx % (FRAME_SKIP + 1)) != 0:
            out.write(frame)
            frame_idx += 1
            continue

        small_frame = cv2.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
        outputs = pose_detector(small_frame)

        if outputs["instances"].has("pred_keypoints") and len(outputs["instances"].pred_keypoints) > 0:
            keypoints = outputs["instances"].pred_keypoints[0].cpu().numpy()
            keypoints[:, :2] /= RESIZE_FACTOR
            draw_keypoints(keypoints, frame)

            keypoints_for_lstm = keypoints[KEYPOINT_INDICES].reshape(-1)
            sequence.append(keypoints_for_lstm)
        else:
            sequence.append(np.zeros(len(KEYPOINT_INDICES) * 3))

        if len(sequence) == SEQUENCE_LENGTH:
            input_np = np.array(sequence, dtype=np.float32)
            input_tensor = torch.from_numpy(input_np).unsqueeze(0).to(cfg.MODEL.DEVICE)
            with torch.no_grad():
                logits = model(input_tensor)
                pred_class = logits.argmax(dim=1).item()
                draw_action_label(frame, CLASS_LABELS[pred_class])

        out.write(frame)
        if USE_IMSHOW:
            cv2.imshow("Action Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

    cap.release()
    out.release()
    if USE_IMSHOW:
        cv2.destroyAllWindows()
    print(f"Processing complete. Output saved as {output_path}")
