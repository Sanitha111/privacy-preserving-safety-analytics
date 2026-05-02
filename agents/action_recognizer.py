# agents/action_recognizer.py — Agent 3: Action Recognizer
import torch
import torch.nn as nn
import numpy as np
from config import ACTIONS, DANGEROUS_ACTIONS, CONFIDENCE_THRESHOLD
from models.stgcn import STGCN  # Use the single canonical STGCN definition


class ActionRecognizerAgent:
    """
    Agent 3 — Action Recognizer
    Uses ST-GCN to classify skeleton sequences into actions.
    num_classes is derived from config.ACTIONS — never hardcoded.
    """
    def __init__(self):
        self.name = "Action Recognizer Agent"
        self.num_classes = len(ACTIONS)
        self.model = STGCN(num_joints=25, num_classes=self.num_classes)
        import os
        model_path = "models/saved_models/stgcn_ntu60 (2).pth"
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
            self.model.eval()
            print("   Weights  : NTU60 trained model loaded ✅")
        self.actions = ACTIONS
        self.dangerous_actions = DANGEROUS_ACTIONS
        print(f"🤖 {self.name} initialized!")
        print(f"   Model    : Spatio-Temporal Graph Convolutional Network")
        print(f"   Classes  : {self.num_classes} → {list(ACTIONS.values())}")

    def train_demo(self, sequences, labels, epochs=20):
        """Quick training for demo purposes"""
        print(f"\n🧠 Training ST-GCN (demo)...")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        from agents.graph_builder import GraphBuilderAgent
        A = GraphBuilderAgent().get_adjacency_matrix()

        losses = []
        self.model.train()

        for epoch in range(epochs):
            epoch_loss = 0
            for seq, label in zip(sequences, labels):
                x = torch.FloatTensor(seq).permute(2, 0, 1).unsqueeze(0)  # (1, C, T, V)
                y = torch.LongTensor([label])

                optimizer.zero_grad()
                output = self.model(x, A)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            losses.append(epoch_loss / len(sequences))
            if (epoch + 1) % 5 == 0:
                print(f"   Epoch [{epoch+1}/{epochs}]  Loss: {losses[-1]:.4f}")

        print("✅ Demo training complete!")
        return losses

    def predict(self, sequence, adjacency_matrix):
        """
        Predict action from a skeleton sequence.
        Returns action label, confidence, and whether it is dangerous.
        """
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(sequence).permute(2, 0, 1).unsqueeze(0)  # (1, C, T, V)
            output = self.model(x, adjacency_matrix)
            probabilities = torch.softmax(output, dim=1)[0]
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()

        action_name = self.actions[predicted_class]
        is_dangerous = predicted_class in self.dangerous_actions

        return {
            "action": action_name,
            "action_id": predicted_class,
            "confidence": confidence,
            "is_dangerous": is_dangerous,
            "all_probabilities": {
                self.actions[i]: float(probabilities[i])
                for i in range(self.num_classes)
            }
        }

    def predict_rule_based(self, sequence):
        """
        Rule-based prediction — works without a trained model.
        Delegates to GraphBuilderAgent.rule_based_detection().
        """
        from agents.graph_builder import GraphBuilderAgent
        result = GraphBuilderAgent().rule_based_detection(sequence)

        # Map graph_builder action names to config ACTIONS names
        action_name_map = {
            "Fall Detected":    "Fall",
            "Collapse":         "Fall",
            "Motionless Alert": "Motionless",
            "Seizure Detected": "Fall",
            "Normal":           "Normal",
        }
        mapped_action = action_name_map.get(result["action"], result["action"])
        is_dangerous = mapped_action in [ACTIONS[i] for i in DANGEROUS_ACTIONS]

        return {
            "action":     mapped_action,
            "confidence": result["confidence"],
            "is_dangerous": is_dangerous,
            "reason":     result["reason"],
            "features":   result["features"],
        }


if __name__ == "__main__":
    from agents.skeleton_extractor import SkeletonExtractorAgent

    extractor  = SkeletonExtractorAgent()
    recognizer = ActionRecognizerAgent()

    fall_seq   = extractor.generate_demo_skeleton_sequence("fall")
    normal_seq = extractor.generate_demo_skeleton_sequence("normal")

    print("\n📊 Testing Rule-Based Prediction:")
    fall_result = recognizer.predict_rule_based(fall_seq)
    print(f"   Fall   → {fall_result['action']} ({fall_result['confidence']:.1%})  dangerous={fall_result['is_dangerous']}")

    normal_result = recognizer.predict_rule_based(normal_seq)
    print(f"   Normal → {normal_result['action']} ({normal_result['confidence']:.1%})  dangerous={normal_result['is_dangerous']}")

    print("\n✅ Agent 3 working correctly!")