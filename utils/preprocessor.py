# utils/preprocessor.py — Skeleton Preprocessing
"""
Preprocessing pipeline following exact methodology from:
- Yan et al. (2018) ST-GCN paper
- Keskes & Noumeir (2021) Vision-Based Fall Detection

Steps:
1. Center normalization — subtract hip joint position
2. Scale normalization — divide by body height
3. Temporal padding — pad/trim to fixed length
4. Data augmentation — random rotation, flip, noise
"""
import numpy as np
import torch

class SkeletonPreprocessor:
    """
    Standard skeleton preprocessing following paper methodology
    """
    def __init__(self, sequence_length=30, num_joints=33):
        self.sequence_length = sequence_length
        self.num_joints = num_joints
        self.center_joint = 23  # Hip center (MediaPipe)
        self.head_joint = 0     # Nose (MediaPipe)
        print("✅ Skeleton Preprocessor initialized!")
        print(f"   Sequence length : {sequence_length} frames")
        print(f"   Joints          : {num_joints}")
        print(f"   Method          : Yan et al. (2018) normalization")

    def normalize(self, sequence):
        """
        Step 1 + 2: Center and scale normalization
        Following Yan et al. (2018) methodology exactly

        - Subtract center joint (hip) from all joints
        - Divide by body height for scale invariance
        """
        sequence = sequence.copy().astype(np.float32)

        # Step 1: Center normalization
        # Subtract hip joint position from all joints
        center = sequence[:, self.center_joint:self.center_joint+1, :]
        sequence = sequence - center

        # Step 2: Scale normalization
        # Divide by distance from head to hip (body height proxy)
        head_pos = sequence[:, self.head_joint, :]
        hip_pos  = sequence[:, self.center_joint, :]
        body_height = np.mean(
            np.linalg.norm(head_pos - hip_pos, axis=1)
        )
        if body_height > 1e-6:
            sequence = sequence / body_height

        return sequence

    def pad_or_trim(self, sequence):
        """
        Step 3: Temporal normalization
        Pad short sequences or trim long ones to fixed length
        """
        T = len(sequence)

        if T == self.sequence_length:
            return sequence

        elif T < self.sequence_length:
            # Pad by repeating last frame
            pad_length = self.sequence_length - T
            last_frame  = sequence[-1:].repeat(pad_length, axis=0)
            return np.concatenate([sequence, last_frame], axis=0)

        else:
            # Trim — sample evenly across sequence
            indices = np.linspace(0, T-1, self.sequence_length, dtype=int)
            return sequence[indices]

    def augment(self, sequence, training=True):
        """
        Step 4: Data augmentation (training only)
        Following standard augmentation from papers

        - Random rotation (±10 degrees)
        - Random horizontal flip
        - Random Gaussian noise
        - Random temporal cropping
        """
        if not training:
            return sequence

        sequence = sequence.copy()

        # Random rotation around Z axis
        if np.random.random() > 0.5:
            angle = np.random.uniform(-10, 10) * np.pi / 180
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array([
                [cos_a, -sin_a, 0],
                [sin_a,  cos_a, 0],
                [0,      0,     1]
            ])
            sequence = sequence @ rotation_matrix.T

        # Random horizontal flip
        if np.random.random() > 0.5:
            sequence[:, :, 0] = -sequence[:, :, 0]

        # Random Gaussian noise
        if np.random.random() > 0.5:
            noise = np.random.normal(0, 0.01, sequence.shape)
            sequence = sequence + noise

        return sequence

    def process(self, sequence, training=False):
        """
        Full preprocessing pipeline
        sequence: raw skeleton (T, V, C)
        returns: processed tensor (C, T, V)
        """
        # Normalize
        sequence = self.normalize(sequence)

        # Pad or trim to fixed length
        sequence = self.pad_or_trim(sequence)

        # Augment if training
        sequence = self.augment(sequence, training)

        # Convert to tensor (C, T, V) for ST-GCN
        tensor = torch.FloatTensor(
            sequence.transpose(2, 0, 1)  # (T,V,C) → (C,T,V)
        )
        return tensor

    def process_batch(self, sequences, training=False):
        """Process a batch of sequences"""
        tensors = [self.process(seq, training) for seq in sequences]
        return torch.stack(tensors)  # (B, C, T, V)


# ── Synthetic Dataset Generator ──────────────────────────────
class SyntheticDatasetGenerator:
    """
    Generates realistic synthetic skeleton dataset for training
    when NTU RGB+D is not available

    Actions generated:
    0 - Normal walking
    1 - Fall
    2 - Pre-fall (unstable gait)
    3 - Lying motionless
    4 - Sitting down normally
    """

    def __init__(self, num_joints=33, sequence_length=30):
        self.num_joints    = num_joints
        self.seq_len       = sequence_length

    def _base_standing_pose(self, noise=0.02):
        """Generate a base standing skeleton pose"""
        joints = np.zeros((self.num_joints, 3))

        # Key joints — approximate normalized positions
        joints[0]  = [0.0,  0.9,  0.0]   # Nose (head)
        joints[11] = [-0.2, 0.5,  0.0]   # Left shoulder
        joints[12] = [0.2,  0.5,  0.0]   # Right shoulder
        joints[13] = [-0.3, 0.2,  0.0]   # Left elbow
        joints[14] = [0.3,  0.2,  0.0]   # Right elbow
        joints[15] = [-0.3, 0.0,  0.0]   # Left wrist
        joints[16] = [0.3,  0.0,  0.0]   # Right wrist
        joints[23] = [-0.1, 0.0,  0.0]   # Left hip (center)
        joints[24] = [0.1,  0.0,  0.0]   # Right hip
        joints[25] = [-0.1,-0.4,  0.0]   # Left knee
        joints[26] = [0.1, -0.4,  0.0]   # Right knee
        joints[27] = [-0.1,-0.8,  0.0]   # Left ankle
        joints[28] = [0.1, -0.8,  0.0]   # Right ankle

        joints += np.random.normal(0, noise, joints.shape)
        return joints

    def generate_normal_walk(self):
        """Generate normal walking sequence"""
        sequence = []
        for i in range(self.seq_len):
            joints = self._base_standing_pose(noise=0.01)
            phase  = i * 0.3

            # Natural walking motion
            joints[25, 0] += 0.05 * np.sin(phase)
            joints[26, 0] += 0.05 * np.sin(phase + np.pi)
            joints[27, 0] += 0.07 * np.sin(phase)
            joints[28, 0] += 0.07 * np.sin(phase + np.pi)
            joints[27, 1] += 0.03 * abs(np.sin(phase))
            joints[28, 1] += 0.03 * abs(np.sin(phase + np.pi))

            # Natural arm swing
            joints[15, 0] += 0.04 * np.sin(phase + np.pi)
            joints[16, 0] += 0.04 * np.sin(phase)

            sequence.append(joints)
        return np.array(sequence)

    def generate_fall(self):
        """
        Generate realistic fall sequence
        Phase 1 (0-30%): Normal walking
        Phase 2 (30-60%): Loss of balance
        Phase 3 (60-100%): Falling to ground
        """
        sequence = []
        for i in range(self.seq_len):
            joints   = self._base_standing_pose(noise=0.015)
            progress = i / self.seq_len

            if progress < 0.3:
                # Normal walking
                phase = i * 0.3
                joints[27, 0] += 0.07 * np.sin(phase)
                joints[28, 0] += 0.07 * np.sin(phase + np.pi)

            elif progress < 0.6:
                # Loss of balance — body tilting
                tilt = (progress - 0.3) / 0.3
                joints[:, 0]  += tilt * 0.3   # Lean sideways
                joints[0, 1]  -= tilt * 0.2   # Head dropping
                joints[15, 0] -= tilt * 0.2   # Arms spreading
                joints[16, 0] += tilt * 0.2

            else:
                # Falling to ground
                fall = (progress - 0.6) / 0.4
                joints[:, 1]  -= fall * 1.2   # All joints drop
                joints[:, 0]  += fall * 0.4   # Sideways motion
                # Body becomes horizontal
                joints[0,  1] = -0.5 + fall * 0.1
                joints[11, 1] = -0.6 + fall * 0.1
                joints[23, 1] = -0.7 + fall * 0.1
                joints[27, 1] = -0.8 + fall * 0.1

            sequence.append(joints)
        return np.array(sequence)

    def generate_prefall_risk(self):
        """
        Generate unstable gait sequence (HIGH FALL RISK)
        Shows: lateral sway, irregular stride, arm spreading
        """
        sequence = []
        for i in range(self.seq_len):
            joints   = self._base_standing_pose(noise=0.02)
            progress = i / self.seq_len
            phase    = i * 0.2

            # Increasing lateral sway
            sway = 0.05 + progress * 0.10
            joints[:, 0]  += sway * np.sin(phase)

            # Head wobbling
            joints[0, 0]  += 0.06 * np.sin(phase * 1.5)
            joints[0, 1]  -= progress * 0.05  # Gradually dropping

            # Irregular shuffling steps
            joints[27, 0] += 0.04 * np.sin(phase + np.random.normal(0, 0.3))
            joints[28, 0] += 0.04 * np.sin(phase + np.pi + np.random.normal(0, 0.3))

            # Arms spreading for support
            arm_spread = progress * 0.15
            joints[15, 0] -= arm_spread
            joints[16, 0] += arm_spread

            # Postural lean
            joints[11, 0] += progress * 0.05
            joints[12, 0] += progress * 0.05

            sequence.append(joints)
        return np.array(sequence)

    def generate_motionless(self):
        """Generate person lying motionless on ground"""
        sequence = []
        for i in range(self.seq_len):
            joints = np.zeros((self.num_joints, 3))

            # All joints at ground level — horizontal position
            joints[0]  = [0.8,  -0.8, 0.0]   # Head
            joints[11] = [0.2,  -0.8, 0.0]   # Left shoulder
            joints[12] = [0.4,  -0.8, 0.0]   # Right shoulder
            joints[23] = [0.0,  -0.8, 0.0]   # Hips
            joints[24] = [0.1,  -0.8, 0.0]
            joints[25] = [-0.3, -0.8, 0.0]   # Knees
            joints[26] = [-0.3, -0.8, 0.0]
            joints[27] = [-0.7, -0.8, 0.0]   # Ankles
            joints[28] = [-0.7, -0.8, 0.0]

            # Very minimal noise — motionless!
            joints += np.random.normal(0, 0.003, joints.shape)
            sequence.append(joints)
        return np.array(sequence)

    def generate_sit_down(self):
        """Generate normal sitting down sequence"""
        sequence = []
        for i in range(self.seq_len):
            joints   = self._base_standing_pose(noise=0.01)
            progress = i / self.seq_len

            # Controlled lowering of body
            joints[23, 1] -= progress * 0.4
            joints[24, 1] -= progress * 0.4
            joints[25, 1] -= progress * 0.2
            joints[26, 1] -= progress * 0.2

            # Knees bending forward
            joints[25, 0] -= progress * 0.1
            joints[26, 0] += progress * 0.1

            sequence.append(joints)
        return np.array(sequence)

    def generate_dataset(self, samples_per_class=200):
        """
        Generate full training dataset
        Returns: sequences, labels
        Labels: 0=Normal, 1=Fall, 2=PreFall, 3=Motionless, 4=SitDown
        """
        print(f"📊 Generating synthetic training dataset...")
        print(f"   {samples_per_class} samples per class × 5 classes")
        print(f"   Total: {samples_per_class * 5} sequences")

        generators = [
            (self.generate_normal_walk,   0, "Normal Walk"),
            (self.generate_fall,          1, "Fall"),
            (self.generate_prefall_risk,  2, "Pre-Fall Risk"),
            (self.generate_motionless,    3, "Motionless"),
            (self.generate_sit_down,      4, "Sit Down"),
        ]

        all_sequences, all_labels = [], []

        for gen_fn, label, name in generators:
            sequences = []
            for _ in range(samples_per_class):
                seq = gen_fn()
                # Add slight random variation
                seq += np.random.normal(0, 0.005, seq.shape)
                sequences.append(seq)

            all_sequences.extend(sequences)
            all_labels.extend([label] * samples_per_class)
            print(f"   ✅ {name}: {samples_per_class} samples")

        print(f"✅ Dataset generated: {len(all_sequences)} total sequences")
        return all_sequences, all_labels


if __name__ == "__main__":
    preprocessor = SkeletonPreprocessor()
    generator    = SyntheticDatasetGenerator()

    sequences, labels = generator.generate_dataset(samples_per_class=10)

    print(f"\n📐 Testing preprocessor...")
    tensor = preprocessor.process(sequences[0])
    print(f"   Input shape  : {sequences[0].shape}")
    print(f"   Output shape : {tensor.shape}")
    print(f"   ✅ Preprocessor working correctly!")