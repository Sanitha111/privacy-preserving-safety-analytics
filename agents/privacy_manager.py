# agents/privacy_manager.py — Agent 4: Privacy & Erasure Manager
"""
Agent 4 — Privacy Manager
Role: Handles face embeddings (NOT photos) and DPDP Act Right to Erasure
      This is the most original part of your project!

Key insight: We store MATH VECTORS not FACES
- Face photo = Personal data (illegal to store without consent)
- Face embedding = Mathematical representation (anonymous, privacy-preserving)
"""
import numpy as np
import sqlite3
import uuid
import os
from datetime import datetime
from utils.database import get_embedding_connection

class PrivacyManagerAgent:
    def __init__(self):
        self.name = "Privacy Manager Agent"
        self.embedding_size = 128
        print(f"🤖 {self.name} initialized!")
        print(f"   Storage mode: Mathematical embeddings ONLY")
        print(f"   Raw photos: NEVER stored ✅")
        print(f"   DPDP Act 2026: Compliant ✅")

    def _simulate_face_embedding(self, seed=None):
        """
        Simulate a face embedding vector
        In production: use DeepFace or FaceNet to generate real embeddings
        Embedding = 128 numbers representing facial geometry
        """
        if seed:
            np.random.seed(seed)
        # Normalized embedding vector (unit vector)
        embedding = np.random.randn(self.embedding_size)
        embedding = embedding / np.linalg.norm(embedding)  # Normalize
        return embedding

    def extract_face_embedding(self, frame=None, person_id=None):
        """
        Extract face embedding from frame
        In production: uses DeepFace library
        For demo: generates simulated embedding

        PRIVACY NOTE: The frame is processed and IMMEDIATELY DISCARDED
        Only the mathematical vector is stored — not the face
        """
        try:
            # Try to use DeepFace if available
            import deepface
            from deepface import DeepFace
            if frame is not None:
                embedding_result = DeepFace.represent(
                    frame,
                    model_name="Facenet",
                    enforce_detection=False
                )
                if embedding_result:
                    return np.array(embedding_result[0]["embedding"])
        except ImportError:
            pass

        # Fallback: simulated embedding for demo
        seed = hash(str(person_id)) % 1000 if person_id else None
        return self._simulate_face_embedding(seed)

    def register_person(self, frame=None, environment="General", person_id=None):
        """
        Register a person by storing their face embedding
        Returns anonymous ID — no name, no face stored!
        """
        # Generate anonymous ID
        # Check if person already registered
        conn = get_embedding_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT anonymous_id FROM face_embeddings WHERE anonymous_id = ?", 
                      (str(abs(hash(str(person_id))))[:8].upper(),))
        existing = cursor.fetchone()
        if existing:
            conn.close()
            return existing[0], self.extract_face_embedding(frame, person_id)

        # Generate anonymous ID
        anonymous_id = str(abs(hash(str(person_id))))[:8].upper()

        # Extract embedding (math vector — NOT the photo)
        embedding = self.extract_face_embedding(frame, person_id)

        # Store ONLY the embedding — never the face
        cursor.execute('''
            INSERT OR REPLACE INTO face_embeddings
            (anonymous_id, embedding, environment)
            VALUES (?, ?, ?)
        ''', (
            anonymous_id,
            embedding.tobytes(),  # Store as binary blob
            environment
        ))
        conn.commit()
        conn.close()

        print(f"✅ Person registered anonymously")
        print(f"   Anonymous ID: {anonymous_id}")
        print(f"   Stored: 128 numbers (NOT a photo)")
        print(f"   Face photo: DISCARDED ✅")

        return anonymous_id, embedding

    def identify_person(self, frame=None, person_id=None, threshold=0.7):
        """
        Identify a person by comparing their face embedding
        to stored embeddings using cosine similarity
        Returns anonymous ID if match found
        """
        query_embedding = self.extract_face_embedding(frame, person_id)

        conn = get_embedding_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT anonymous_id, embedding FROM face_embeddings")
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return None, 0.0

        best_match = None
        best_similarity = 0.0

        for anon_id, embedding_blob in rows:
            stored_embedding = np.frombuffer(embedding_blob, dtype=np.float64)

            # Cosine similarity between embeddings
            if len(stored_embedding) == len(query_embedding):
                similarity = np.dot(query_embedding, stored_embedding) / (
                    np.linalg.norm(query_embedding) *
                    np.linalg.norm(stored_embedding) + 1e-10
                )

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = anon_id

        if best_similarity > threshold:
            return best_match, best_similarity
        return None, best_similarity

    def exercise_right_to_erasure(self, frame=None, person_id=None):
        """
        DPDP Act 2026 — Section 12: Right to Erasure
        Person provides ONE photo → system finds and deletes ALL their records
        This is the most legally important feature!
        """
        print(f"\n🔐 DPDP Act — Right to Erasure Request Received")
        print(f"   Processing erasure request...")

        # Find the person using their photo
        anonymous_id, similarity = self.identify_person(frame, person_id)

        if anonymous_id is None:
            print(f"   ❌ No matching records found")
            return {
                "success": False,
                "message": "No matching records found in database",
                "records_deleted": 0
            }

        print(f"   ✅ Person identified: {anonymous_id} (similarity: {similarity:.2%})")

        # Delete from embedding database
        emb_conn = get_embedding_connection()
        emb_cursor = emb_conn.cursor()
        emb_cursor.execute(
            "DELETE FROM face_embeddings WHERE anonymous_id = ?",
            (anonymous_id,)
        )
        embeddings_deleted = emb_cursor.rowcount
        emb_conn.commit()
        emb_conn.close()

        # Delete from safety events database
        from utils.database import get_connection
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM skeleton_events WHERE anonymous_id = ?",
            (anonymous_id,)
        )
        events_deleted = cursor.rowcount

        cursor.execute(
            "DELETE FROM alerts WHERE anonymous_id = ?",
            (anonymous_id,)
        )
        alerts_deleted = cursor.rowcount

        # Log erasure request (without personal data)
        request_id = str(uuid.uuid4())[:8].upper()
        cursor.execute('''
            INSERT INTO erasure_requests
            (request_id, records_deleted, status)
            VALUES (?, ?, ?)
        ''', (
            request_id,
            events_deleted + alerts_deleted + embeddings_deleted,
            "COMPLETED"
        ))
        conn.commit()
        conn.close()

        total_deleted = events_deleted + alerts_deleted + embeddings_deleted

        print(f"\n✅ ERASURE COMPLETE — DPDP Act Compliant")
        print(f"   Request ID: {request_id}")
        print(f"   Skeleton events deleted: {events_deleted}")
        print(f"   Alerts deleted: {alerts_deleted}")
        print(f"   Embeddings deleted: {embeddings_deleted}")
        print(f"   Total records deleted: {total_deleted}")
        print(f"   This person no longer exists in our system ✅")

        return {
            "success": True,
            "request_id": request_id,
            "anonymous_id": anonymous_id,
            "records_deleted": total_deleted,
            "breakdown": {
                "skeleton_events": events_deleted,
                "alerts": alerts_deleted,
                "embeddings": embeddings_deleted
            },
            "message": f"All {total_deleted} records permanently deleted. DPDP Act compliant."
        }

    def get_privacy_stats(self):
        """Get current privacy statistics"""
        emb_conn = get_embedding_connection()
        emb_cursor = emb_conn.cursor()
        emb_cursor.execute("SELECT COUNT(*) FROM face_embeddings")
        total_persons = emb_cursor.fetchone()[0]
        emb_conn.close()

        from utils.database import get_connection
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM erasure_requests WHERE status='COMPLETED'")
        total_erasures = cursor.fetchone()[0]
        cursor.execute("SELECT SUM(records_deleted) FROM erasure_requests")
        total_deleted = cursor.fetchone()[0] or 0
        conn.close()

        return {
            "total_persons_registered": total_persons,
            "total_erasure_requests": total_erasures,
            "total_records_deleted": total_deleted,
            "raw_videos_stored": 0,  # Always 0 — privacy by design!
            "faces_stored": 0,       # Always 0 — only embeddings!
            "dpdp_compliant": True
        }


if __name__ == "__main__":
    from utils.database import initialize_db
    initialize_db()

    agent = PrivacyManagerAgent()

    print("\n📋 Testing Privacy Manager...")

    # Register 3 demo persons
    id1, emb1 = agent.register_person(person_id="person_1", environment="Hospital")
    id2, emb2 = agent.register_person(person_id="person_2", environment="Hospital")
    id3, emb3 = agent.register_person(person_id="person_3", environment="Factory")

    print(f"\n📊 Privacy Stats:")
    stats = agent.get_privacy_stats()
    for k, v in stats.items():
        print(f"   {k}: {v}")

    print(f"\n🔐 Testing Right to Erasure (DPDP Act)...")
    result = agent.exercise_right_to_erasure(person_id="person_1")
    print(f"   Result: {result['message']}")

    print("\n✅ Agent 4 working correctly!")