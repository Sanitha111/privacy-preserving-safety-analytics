# agents/skeleton_extractor.py — Agent 1: Privacy Filter
import cv2
import numpy as np
import os
from datetime import datetime
from config import NUM_JOINTS, SKELETON_PATH

os.makedirs(SKELETON_PATH, exist_ok=True)

# Safely import mediapipe
try:
    import mediapipe as mp
    if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'pose'):
        _MP_AVAILABLE = True
    else:
        _MP_AVAILABLE = False
except Exception:
    _MP_AVAILABLE = False

class SkeletonExtractorAgent:
    def __init__(self):
        self.name = "Skeleton Extractor Agent"
        self.pose     = None
        self.mp_pose  = None
        self.mp_draw  = None

        if _MP_AVAILABLE:
            try:
                self.mp_pose = mp.solutions.pose
                self.mp_draw = mp.solutions.drawing_utils
                self.pose    = self.mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    smooth_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                print(f"🤖 {self.name} initialized! (MediaPipe mode ✅)")
            except Exception as e:
                print(f"🤖 {self.name} initialized! (Synthetic mode — {e})")
        else:
            print(f"🤖 {self.name} initialized! (Synthetic mode ✅)")
        print(f"   Privacy Mode: ON — No faces or raw video stored")

    def extract_skeleton(self, frame):
        if self.pose is None:
            return None, None
        try:
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res   = self.pose.process(rgb)
            if res.pose_landmarks:
                skel = [[l.x, l.y, l.z, l.visibility]
                        for l in res.pose_landmarks.landmark]
                return np.array(skel), res.pose_landmarks
        except Exception:
            pass
        return None, None

    def draw_skeleton_only(self, frame, landmarks):
        blank = np.zeros(frame.shape, dtype=np.uint8)
        if landmarks and self.mp_pose and self.mp_draw:
            try:
                self.mp_draw.draw_landmarks(
                    blank, landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_draw.DrawingSpec(
                        color=(0,255,0), thickness=3, circle_radius=4),
                    connection_drawing_spec=self.mp_draw.DrawingSpec(
                        color=(0,200,255), thickness=2))
            except Exception:
                pass
        return blank

    def generate_demo_skeleton_sequence(self, action="fall", n_frames=30):
        """Generate synthetic skeleton sequence — works without camera"""
        np.random.seed(abs(hash(action)) % 1000)
        sequence = []

        for i in range(n_frames):
            joints   = np.zeros((33, 3))
            progress = i / n_frames

            # Base standing pose
            joints[0]  = [0.5,  0.9, 0.0]
            joints[11] = [0.4,  0.7, 0.0]
            joints[12] = [0.6,  0.7, 0.0]
            joints[13] = [0.35, 0.55,0.0]
            joints[14] = [0.65, 0.55,0.0]
            joints[15] = [0.3,  0.4, 0.0]
            joints[16] = [0.7,  0.4, 0.0]
            joints[23] = [0.45, 0.5, 0.0]
            joints[24] = [0.55, 0.5, 0.0]
            joints[25] = [0.45, 0.3, 0.0]
            joints[26] = [0.55, 0.3, 0.0]
            joints[27] = [0.45, 0.1, 0.0]
            joints[28] = [0.55, 0.1, 0.0]

            if action == "fall":
                if progress > 0.3:
                    fall = (progress - 0.3) / 0.7
                    joints[:, 1] -= fall * 0.6
                    joints[:, 0] += fall * 0.2
                    joints[0, 1]  = max(0.05, 0.9 - fall * 0.85)

            elif action == "normal":
                phase = i * 0.3
                joints[27,0] += 0.03 * np.sin(phase)
                joints[28,0] += 0.03 * np.sin(phase + np.pi)
                joints[25,1] += 0.02 * abs(np.sin(phase))
                joints[26,1] += 0.02 * abs(np.sin(phase + np.pi))

            elif action == "motionless":
                joints[:, 1] -= 0.5
                joints[:, 0] += 0.15

            elif action == "prefall":
                sway = 0.03 + progress * 0.08
                joints[:, 0]  += sway * np.sin(i * 0.2)
                joints[0,  1] -= progress * 0.03
                joints[15, 0] -= progress * 0.1
                joints[16, 0] += progress * 0.1

            joints += np.random.normal(0, 0.008, (33, 3))
            sequence.append(joints)

        return np.array(sequence)

    def extract_from_webcam(self, duration_seconds=10, environment="General"):
        if self.pose is None:
            print("⚠️ MediaPipe unavailable — using synthetic data")
            return [self.generate_demo_skeleton_sequence(a)
                    for a in ["normal", "fall", "normal"]]

        print(f"📡 Starting webcam ({duration_seconds}s)...")
        cap, sequences, current = cv2.VideoCapture(0), [], []
        start = datetime.now()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or (datetime.now()-start).seconds >= duration_seconds:
                break
            skel, lm = self.extract_skeleton(frame)
            if skel is not None:
                current.append(skel[:, :3])
                sf = self.draw_skeleton_only(frame, lm)
                cv2.putText(sf, "GHOST-VISION | Privacy: ON",
                           (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.imshow("Ghost-Vision", sf)
            if len(current) == 30:
                sequences.append(np.array(current))
                current = []
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print(f"✅ Extracted {len(sequences)} sequences")
        return sequences


if __name__ == "__main__":
    agent = SkeletonExtractorAgent()
    for action in ["normal", "fall", "motionless", "prefall"]:
        seq = agent.generate_demo_skeleton_sequence(action)
        print(f"   {action}: {seq.shape} ✅")