# agents/orchestrator.py — LLM Orchestrator Agent
"""
The TRUE Agentic component of Ghost-Vision
Uses Gemini LLM to dynamically reason and decide
which agents to invoke and in what order
"""
import requests
import json
import os
from dotenv import load_dotenv
load_dotenv()

class OrchestratorAgent:
    def __init__(self):
        self.name = "LLM Orchestrator Agent"
        self.api_key = os.getenv("GEMINI_API_KEY", "")
        self.decision_history = []
        print(f"🤖 {self.name} initialized!")
        print(f"   API: {'Connected' if self.api_key else 'No key — using fallback'}")

    def _call_gemini(self, prompt):
        if not self.api_key:
            return None
        try:
            url = (f"https://generativelanguage.googleapis.com/v1beta/"
       f"models/gemini-2.0-flash:generateContent?key={self.api_key}")
        
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.2, "maxOutputTokens": 200}
            }
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                return response.json()["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            pass
        return None

    def reason_and_decide(self, prefall_risk, stgcn_action,
                          stgcn_confidence, severity, environment):
        prompt = f"""You are a hospital safety AI orchestrator.
Current sensor readings:
- Pre-Fall Risk Score (LSTM): {prefall_risk:.1%}
- Detected Action (ST-GCN): {stgcn_action}
- Detection Confidence: {stgcn_confidence:.1%}
- Fall Severity (CNN): {severity if severity else 'N/A'}
- Environment: {environment}

Decide ONE of:
1. IMMEDIATE_ALERT
2. MONITOR_CLOSELY
3. MILD_WARNING
4. CONTINUE_NORMAL

Respond ONLY with JSON: {{"decision": "IMMEDIATE_ALERT", "reason": "brief reason", "urgency": 9}}"""

        response = self._call_gemini(prompt)
        if response:
            try:
                clean = response.strip()
                if "```" in clean:
                    clean = clean.split("```")[1]
                    if clean.startswith("json"):
                        clean = clean[4:]
                result = json.loads(clean.strip())
                result["source"] = "LLM (Gemini)"
                self.decision_history.append(result)
                return result
            except Exception:
                pass
        result = self._fallback_decision(prefall_risk, stgcn_action, stgcn_confidence)
        self.decision_history.append(result)
        return result

    def _fallback_decision(self, risk, action, confidence):
        if action in ["Fall", "Motionless"] and confidence > 0.85:
            return {"decision": "IMMEDIATE_ALERT", "reason": "Fall confirmed with high confidence", "urgency": 10, "source": "Rule-based"}
        elif action in ["Fall", "Motionless"] and confidence > 0.60:
            return {"decision": "IMMEDIATE_ALERT", "reason": "Fall detected — moderate confidence", "urgency": 8, "source": "Rule-based"}
        elif risk > 0.70:
            return {"decision": "MONITOR_CLOSELY", "reason": f"High pre-fall risk: {risk:.1%}", "urgency": 7, "source": "Rule-based"}
        elif risk > 0.45:
            return {"decision": "MILD_WARNING", "reason": f"Moderate instability: {risk:.1%}", "urgency": 4, "source": "Rule-based"}
        else:
            return {"decision": "CONTINUE_NORMAL", "reason": "No anomaly detected", "urgency": 1, "source": "Rule-based"}

    def reflect_and_adapt(self):
        if len(self.decision_history) < 5:
            return "Not enough history to reflect yet"
        recent = self.decision_history[-5:]
        alert_count = sum(1 for d in recent if d.get("decision") == "IMMEDIATE_ALERT")
        prompt = f"""Reviewing last 5 safety decisions: {json.dumps(recent)}
{alert_count}/5 were IMMEDIATE_ALERTs. Should system: A) Keep sensitivity B) Increase C) Decrease?
Respond: ONE letter + ONE sentence."""
        response = self._call_gemini(prompt)
        return response.strip() if response else "Maintaining current sensitivity"

    def get_decision_summary(self):
        if not self.decision_history:
            return {}
        decisions = [d.get("decision") for d in self.decision_history]
        return {
            "total_decisions": len(decisions),
            "immediate_alerts": decisions.count("IMMEDIATE_ALERT"),
            "monitor_closely": decisions.count("MONITOR_CLOSELY"),
            "mild_warnings": decisions.count("MILD_WARNING"),
            "continue_normal": decisions.count("CONTINUE_NORMAL"),
            "llm_powered": sum(1 for d in self.decision_history if "LLM" in d.get("source", ""))
        }