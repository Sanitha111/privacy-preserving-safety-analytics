# main.py — Ghost-Vision Complete Pipeline
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from dotenv import load_dotenv
load_dotenv()

from utils.database import initialize_db
from utils.preprocessor import SkeletonPreprocessor, SyntheticDatasetGenerator
from models.stgcn import STGCN, FallSeverityCNN, PreFallLSTM
from models.train_ntu import STGCN_PATH, SEVERITY_PATH, PREFALL_PATH  # ← import from train_ntu
from agents.graph_builder import GraphBuilderAgent
from agents.skeleton_extractor import SkeletonExtractorAgent
from agents.privacy_manager import PrivacyManagerAgent
from agents.alert_generator import AlertGeneratorAgent
from agents.orchestrator import OrchestratorAgent
from utils.ntu_loader import build_ntu_adjacency, NUM_NTU_JOINTS      # ← import NTU helpers
from config import ACTIONS, DANGEROUS_ACTIONS

ACTION_LABELS   = ACTIONS
SEVERITY_LABELS = {0: "Minor", 1: "Moderate", 2: "Critical"}


def load_or_train_models():
    """Load NTU-trained models or train from scratch"""
    # ↓ use NTU adjacency (25 joints) instead of graph_builder's 33-joint version
    A            = build_ntu_adjacency()

    models_exist = all([
        os.path.exists(STGCN_PATH),
        os.path.exists(SEVERITY_PATH),
        os.path.exists(PREFALL_PATH)
    ])

    if models_exist:
        print("📦 Loading NTU-trained models...")
        stgcn_model    = STGCN(num_joints=NUM_NTU_JOINTS, num_classes=len(ACTIONS))
        severity_model = FallSeverityCNN(num_joints=NUM_NTU_JOINTS)
        prefall_model  = PreFallLSTM(input_size=75)   # 25 * 3 = 75
        stgcn_model.load_state_dict(torch.load(STGCN_PATH,    weights_only=True, map_location=torch.device('cpu')))
        severity_model.load_state_dict(torch.load(SEVERITY_PATH, weights_only=True, map_location=torch.device('cpu')))
        prefall_model.load_state_dict(torch.load(PREFALL_PATH,  weights_only=True, map_location=torch.device('cpu')))
        print("✅ All models loaded!")
    else:
        print("🔄 No saved models found — run: python models/train_ntu.py first")
        sys.exit(1)   # don't silently fall back to synthetic training

    stgcn_model.eval()
    severity_model.eval()
    prefall_model.eval()
    return stgcn_model, severity_model, prefall_model, A


# rest of main.py stays the same ...
def analyze_sequence(sequence, stgcn_model, severity_model,
                     prefall_model, A, preprocessor):
    """Full pure-AI analysis pipeline"""
    tensor = preprocessor.process(sequence).unsqueeze(0)
    tensor = tensor[:, :, :, :25]  # trim to 25 joints for NTU models
    with torch.no_grad():
        # Step 1: Pre-Fall Risk (LSTM)
        flat_seq = sequence[:, :25, :].reshape(sequence.shape[0], -1)  # still correct, auto-sizes
        lstm_input = torch.FloatTensor(flat_seq).unsqueeze(0)
        risk_score = prefall_model(lstm_input).item()

        # Step 2: Fall Detection (ST-GCN)
        stgcn_output = stgcn_model(tensor, A)
        probs        = torch.softmax(stgcn_output, dim=1)[0]
        action_id    = torch.argmax(probs).item()
        confidence   = probs[action_id].item()
        action_name  = ACTION_LABELS[action_id]

        # Step 3: Severity Classification (CNN)
        severity_name = None
        if action_id in [1, 3]:
            sev_output    = severity_model(tensor)
            severity_id   = torch.argmax(sev_output, dim=1).item()
            severity_name = SEVERITY_LABELS[severity_id]

    return {
        "prefall_risk_score": round(risk_score, 3),
        "prefall_risk_pct":   f"{risk_score:.1%}",
        "action":             action_name,
        "action_id":          action_id,
        "confidence":         round(confidence, 3),
        "is_fall":            action_id in [1, 3],
        "severity":           severity_name,
        "all_probs":          {ACTION_LABELS[i]: float(probs[i]) for i in range(len(ACTION_LABELS))}
    }

def run_pipeline():
    print("\n" + "="*65)
    print("👁️  GHOST-VISION — Privacy-Preserving Safety Analytics")
    print("     ST-GCN + CNN + LSTM + LLM Orchestrator")
    print("     Following: Yan et al. (2018) + Novel Extensions")
    print("="*65 + "\n")

    # Initialize
    initialize_db()
    privacy_agent     = PrivacyManagerAgent()
    alert_agent       = AlertGeneratorAgent()
    orchestrator      = OrchestratorAgent()
    generator         = SyntheticDatasetGenerator()

    # Load or train models
    stgcn_model, severity_model, prefall_model, A = load_or_train_models()
    preprocessor = SkeletonPreprocessor()

    environment = "Hospital"
    print(f"\n🏥 Environment: {environment}")
    print("-"*65)

    # Test scenarios
    scenarios = [
        ("Normal walk",       generator.generate_normal_walk),
        ("Unstable gait",     generator.generate_prefall_risk),
        ("Fall",              generator.generate_fall),
        ("Motionless",        generator.generate_motionless),
        ("Sit down",          generator.generate_sit_down),
    ]

    for description, gen_fn in scenarios:
        print(f"\n📹 Scenario: {description}")

        # Generate skeleton sequence
        sequence = gen_fn()
        sequence += np.random.normal(0, 0.005, sequence.shape)
        print(f"   Skeleton: {sequence.shape} | Raw video: DISCARDED ✅")

        # Pure AI analysis
        result = analyze_sequence(
            sequence, stgcn_model, severity_model,
            prefall_model, A, preprocessor
        )
        print(f"   Pre-Fall Risk  : {result['prefall_risk_pct']} "
              f"{'⚠️ HIGH' if result['prefall_risk_score'] > 0.5 else '✅ LOW'}")
        print(f"   ST-GCN Action  : {result['action']} ({result['confidence']:.1%})")
        if result["severity"]:
            print(f"   Severity (CNN) : {result['severity']}")

        # LLM Orchestrator decides what to do
        decision = orchestrator.reason_and_decide(
            prefall_risk     = result["prefall_risk_score"],
            stgcn_action     = result["action"],
            stgcn_confidence = result["confidence"],
            severity         = result["severity"],
            environment      = environment
        )
        print(f"   🤖 Orchestrator: {decision['decision']} "
              f"(urgency: {decision['urgency']}/10) [{decision['source']}]")
        print(f"   💬 Reason      : {decision['reason']}")

        # Privacy + Alert
        anon_id, _ = privacy_agent.register_person(
            person_id=f"demo_{description[:8]}",
            environment=environment
        )
        alert_agent.process_detection(
            {"action": result["action"],
             "confidence": result["confidence"],
             "is_dangerous": result["is_fall"]},
            anon_id, environment
        )

    # Orchestrator reflection
    print("\n" + "-"*65)
    print("🤔 Orchestrator Reflection (Agentic Adaptation):")
    reflection = orchestrator.reflect_and_adapt()
    print(f"   {reflection}")

    # Decision summary
    summary = orchestrator.get_decision_summary()
    print(f"\n📊 Orchestrator Summary:")
    for k, v in summary.items():
        print(f"   {k}: {v}")

    # Privacy stats
    stats         = alert_agent.get_event_statistics()
    privacy_stats = privacy_agent.get_privacy_stats()
    print(f"\n🔐 Privacy Stats:")
    print(f"   Total events     : {stats.get('total_events', 0)}")
    print(f"   Dangerous events : {stats.get('dangerous_events', 0)}")
    print(f"   Raw videos stored: {privacy_stats['raw_videos_stored']} ✅")
    print(f"   Faces stored     : {privacy_stats['faces_stored']} ✅")
    print(f"   DPDP Compliant   : {privacy_stats['dpdp_compliant']} ✅")

    print("\n" + "="*65)
    print("✅ GHOST-VISION PIPELINE COMPLETE!")
    print("="*65)
    print("\n🚀 Launch dashboard: streamlit run dashboard/app.py\n")

if __name__ == "__main__":
    run_pipeline()