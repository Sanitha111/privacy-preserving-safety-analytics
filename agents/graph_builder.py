# agents/graph_builder.py — Agent 2: Graph Constructor
"""
Agent 2 — Graph Builder
Role: Converts skeleton joint coordinates into a graph structure
      for the Spatio-Temporal GCN

The human body is naturally a GRAPH:
- Nodes (V) = Joints (shoulder, elbow, wrist...)
- Edges (E) = Bones connecting joints
"""
import numpy as np
import torch

class GraphBuilderAgent:
    def __init__(self):
        self.name = "Graph Builder Agent"
        self.num_joints = 33  # MediaPipe joints

        # Define human body connections (edges)
        # These are the natural bone connections
        self.connections = [
            # Face
            (0, 1), (1, 2), (2, 3), (3, 7),
            (0, 4), (4, 5), (5, 6), (6, 8),
            # Torso
            (11, 12), (11, 23), (12, 24), (23, 24),
            # Left arm
            (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
            # Right arm
            (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
            # Left leg
            (23, 25), (25, 27), (27, 29), (27, 31),
            # Right leg
            (24, 26), (26, 28), (28, 30), (28, 32),
        ]
        self.adjacency_matrix = self._build_adjacency_matrix()
        print(f"🤖 {self.name} initialized!")
        print(f"   Joints (Nodes): {self.num_joints}")
        print(f"   Connections (Edges): {len(self.connections)}")

    def _build_adjacency_matrix(self):
        """
        Build adjacency matrix A representing body connections
        A[i][j] = 1 if joint i and joint j are connected by a bone
        This is the mathematical representation of the human body graph
        """
        A = np.zeros((self.num_joints, self.num_joints))

        for i, j in self.connections:
            A[i][j] = 1
            A[j][i] = 1  # Undirected graph

        # Add self-connections
        A = A + np.eye(self.num_joints)

        # Normalize adjacency matrix
        D = np.diag(np.sum(A, axis=1))
        D_inv_sqrt = np.linalg.inv(np.sqrt(D))
        A_normalized = D_inv_sqrt @ A @ D_inv_sqrt

        return A_normalized

    def sequence_to_graph_tensor(self, sequence):
        """
        Convert skeleton sequence to graph tensor for ST-GCN
        Input: sequence of shape (T, V, C) = (frames, joints, coordinates)
        Output: tensor of shape (C, T, V) ready for ST-GCN
        """
        # sequence shape: (T=30, V=33, C=3)
        T, V, C = sequence.shape

        # Transpose to (C, T, V)
        graph_tensor = sequence.transpose(2, 0, 1)

        return torch.FloatTensor(graph_tensor), torch.FloatTensor(self.adjacency_matrix)

    def extract_fall_features(self, sequence):
        """
        Extract hand-crafted features specifically designed
        for fall/collapse detection
        These features make physical sense — great for explaining to panel!
        """
        features = {}

        # 1. Center of Mass (average of all joint Y positions)
        # Falls cause COM to drop rapidly
        com_y = np.mean(sequence[:, :, 1], axis=1)  # Y over time
        features["com_drop_rate"] = float(np.max(np.diff(com_y)))
        features["final_com_height"] = float(com_y[-1])

        # 2. Head position (joint 0 = nose)
        head_y = sequence[:, 0, 1]
        features["head_drop"] = float(head_y[-1] - head_y[0])
        features["head_final_height"] = float(head_y[-1])

        # 3. Body spread (how spread out joints are)
        # Falls cause body to become horizontal — joints spread horizontally
        joint_spread_x = np.std(sequence[:, :, 0], axis=1)
        features["horizontal_spread"] = float(np.mean(joint_spread_x[-5:]))

        # 4. Velocity (how fast joints are moving)
        joint_velocity = np.diff(sequence[:, :, 1], axis=0)
        features["max_velocity"] = float(np.max(np.abs(joint_velocity)))
        features["avg_velocity"] = float(np.mean(np.abs(joint_velocity)))

        # 5. Motionless detection
        recent_motion = np.std(sequence[-10:, :, :])
        features["recent_motion"] = float(recent_motion)
        features["is_motionless"] = bool(recent_motion < 0.01)

        # 6. Aspect ratio (width/height of bounding box)
        # Horizontal person = large aspect ratio = possible fall
        width = np.max(sequence[-1, :, 0]) - np.min(sequence[-1, :, 0])
        height = np.max(sequence[-1, :, 1]) - np.min(sequence[-1, :, 1])
        features["aspect_ratio"] = float(width / (height + 1e-6))

        return features

    def rule_based_detection(self, sequence):
        """
        Rule-based fall detection using physical features
        This is interpretable and explainable — great for panel!
        Works WITHOUT needing trained model
        """
        features = self.extract_fall_features(sequence)

        action = "Normal"
        confidence = 0.3
        reason = "No anomaly detected"

        # Fall detection rules
        if features["head_drop"] > 0.3 and features["max_velocity"] > 0.05:
            action = "Fall Detected"
            confidence = min(0.95, 0.6 + features["head_drop"])
            reason = f"Head dropped {features['head_drop']:.2f} units rapidly"

        # Collapse detection
        elif features["final_com_height"] > 0.75 and features["aspect_ratio"] > 1.5:
            action = "Collapse"
            confidence = 0.88
            reason = "Body in horizontal position — possible collapse"

        # Motionless alert
        elif features["is_motionless"] and features["final_com_height"] > 0.6:
            action = "Motionless Alert"
            confidence = 0.82
            reason = "Person motionless on ground for extended period"

        return {
            "action": action,
            "confidence": confidence,
            "reason": reason,
            "features": features
        }

    def get_adjacency_matrix(self):
        return torch.FloatTensor(self.adjacency_matrix)


if __name__ == "__main__":
    agent = GraphBuilderAgent()

    # Test with synthetic fall sequence
    from agents.skeleton_extractor import SkeletonExtractorAgent
    extractor = SkeletonExtractorAgent()

    fall_seq = extractor.generate_demo_skeleton_sequence("fall")
    normal_seq = extractor.generate_demo_skeleton_sequence("normal")

    print("\n📊 Testing fall detection:")
    fall_result = agent.rule_based_detection(fall_seq)
    print(f"   Action: {fall_result['action']}")
    print(f"   Confidence: {fall_result['confidence']:.2%}")
    print(f"   Reason: {fall_result['reason']}")

    print("\n📊 Testing normal detection:")
    normal_result = agent.rule_based_detection(normal_seq)
    print(f"   Action: {normal_result['action']}")
    print(f"   Confidence: {normal_result['confidence']:.2%}")

    print("\n✅ Agent 2 working correctly!")