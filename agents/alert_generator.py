# agents/alert_generator.py — Agent 5: Alert Generator
"""
Agent 5 — Alert Generator
Role: Triggers real-time alerts when dangerous actions detected
      Saves events to database for analysis
"""
import sqlite3
import uuid
from datetime import datetime
from config import DANGEROUS_ACTIONS, CONFIDENCE_THRESHOLD
from utils.database import get_connection

class AlertGeneratorAgent:
    def __init__(self):
        self.name = "Alert Generator Agent"
        self.alert_history = []
        print(f"🤖 {self.name} initialized!")

    def process_detection(self, detection_result, anonymous_id, environment):
        """Process action detection and trigger alert if needed"""
        action = detection_result.get("action", "Normal")
        confidence = detection_result.get("confidence", 0.0)
        is_dangerous = detection_result.get("is_dangerous", False)

        event_id = str(uuid.uuid4())[:8].upper()
        timestamp = datetime.now()

        # Save event to database
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO skeleton_events
            (timestamp, anonymous_id, action_detected,
             confidence, environment, alert_triggered)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            timestamp.isoformat(), anonymous_id, action,
            confidence, environment,
            1 if is_dangerous else 0
        ))

        if is_dangerous and confidence >= CONFIDENCE_THRESHOLD:
            cursor.execute('''
                INSERT INTO alerts
                (timestamp, anonymous_id, action, confidence, environment)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                timestamp.isoformat(), anonymous_id,
                action, confidence, environment
            ))

        conn.commit()
        conn.close()

        alert = {
            "event_id": event_id,
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "action": action,
            "confidence": confidence,
            "is_dangerous": is_dangerous,
            "alert_triggered": is_dangerous and confidence >= CONFIDENCE_THRESHOLD,
            "anonymous_id": anonymous_id,
            "environment": environment,
            "message": self._generate_alert_message(action, confidence, environment)
        }

        if alert["alert_triggered"]:
            self.alert_history.append(alert)
            print(f"\n🚨 ALERT TRIGGERED!")
            print(f"   Action: {action}")
            print(f"   Confidence: {confidence:.1%}")
            print(f"   Environment: {environment}")
            print(f"   Anonymous ID: {anonymous_id}")
            print(f"   Message: {alert['message']}")

        return alert

    def _generate_alert_message(self, action, confidence, environment):
        messages = {
            "Fall Detected": f"URGENT: Person fall detected in {environment}! Immediate assistance required!",
            "Collapse": f"CRITICAL: Person collapse in {environment}! Call emergency services!",
            "Motionless Alert": f"WARNING: Person motionless in {environment}! Check immediately!",
            "Seizure Detected": f"EMERGENCY: Possible seizure in {environment}! Medical help needed!",
            "Restricted Zone Entry": f"SECURITY: Unauthorized entry in {environment}!",
            "Normal": "No anomaly detected. System monitoring normally."
        }
        return messages.get(action, f"Alert: {action} detected in {environment}")

    def get_recent_alerts(self, limit=10):
        """Get recent alerts from database"""
        import pandas as pd
        conn = get_connection()
        try:
            df = pd.read_sql_query(
                f"SELECT * FROM alerts ORDER BY timestamp DESC LIMIT {limit}",
                conn
            )
        except Exception:
            df = pd.DataFrame()
        conn.close()
        return df

    def get_event_statistics(self):
        """Get detection statistics for dashboard"""
        import pandas as pd
        conn = get_connection()
        try:
            events_df = pd.read_sql_query(
                "SELECT * FROM skeleton_events ORDER BY timestamp DESC",
                conn
            )
        except Exception:
            events_df = pd.DataFrame()
        conn.close()

        if len(events_df) == 0:
            return {}

        total = len(events_df)
        dangerous = len(events_df[events_df["alert_triggered"] == 1])

        action_counts = {}
        if "action_detected" in events_df.columns:
            action_counts = events_df["action_detected"].value_counts().to_dict()

        return {
            "total_events": total,
            "dangerous_events": dangerous,
            "normal_events": total - dangerous,
            "detection_rate": dangerous / total if total > 0 else 0,
            "action_counts": action_counts,
            "avg_confidence": float(events_df["confidence"].mean()) if "confidence" in events_df.columns else 0
        }


if __name__ == "__main__":
    from utils.database import initialize_db
    initialize_db()

    agent = AlertGeneratorAgent()

    # Test with fall detection
    fall_detection = {
        "action": "Fall Detected",
        "confidence": 0.92,
        "is_dangerous": True
    }

    result = agent.process_detection(fall_detection, "ABC123", "Hospital")
    print(f"\n✅ Alert test complete!")
    print(f"   Alert triggered: {result['alert_triggered']}")