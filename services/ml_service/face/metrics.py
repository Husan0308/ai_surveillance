from dataclasses import dataclass,asdict
@dataclass
class FaceMetrics:
    face_candidates:int=0;face_attempts:int=0;faces_detected:int=0;faces_rejected_quality:int=0;faces_embedded:int=0
    face_detection_ms:float=0;face_alignment_ms:float=0;face_embedding_gpu_ms:float=0;face_matching_ms:float=0
    known_matches:int=0;ambiguous_matches:int=0;unknown_faces:int=0;identity_conflicts:int=0
    enrollment_good_samples:int=0;enrollment_rejected_samples:int=0
    def snapshot(self):return asdict(self)
