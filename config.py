# config.py — Ghost-Vision Configuration

# --- Paths ---
DB_PATH = "data/ghost_vision.db"
MODEL_PATH = "models/saved_models/stgcn_model.pth"
SCALER_PATH = "models/saved_models/scaler.pkl"
REPORT_PATH = "reports/"
SKELETON_PATH = "data/skeletons/"
EMBEDDING_PATH = "data/embeddings/"

# --- Skeleton Settings ---
NUM_JOINTS = 33
SEQUENCE_LENGTH = 30
CONFIDENCE_THRESHOLD = 0.85

# --- Action Classes (single source of truth for the whole project) ---
ACTIONS = {
    0: "Normal",
    1: "Fall",
    2: "Pre-Fall Risk",
    3: "Motionless",
    4: "Sit Down",
}

# --- Alert Settings ---
DANGEROUS_ACTIONS = [1, 2, 3]   # Fall, Pre-Fall Risk, Motionless
FALL_THRESHOLD = 0.85
MOTIONLESS_THRESHOLD_SECONDS = 30

# --- Privacy Settings ---
FACE_EMBEDDING_SIZE = 128
FAISS_INDEX_PATH = "data/embeddings/face_index.faiss"
EMBEDDING_DB_PATH = "data/embeddings/embeddings.db"

# --- Dashboard Settings ---
CAMERA_INDEX = 0
FPS = 30
FRAME_WIDTH = 640
FRAME_HEIGHT = 480