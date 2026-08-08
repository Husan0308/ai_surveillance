import time,numpy as np
from .gallery import KnownPersonGallery
from .matcher import KnownPersonMatcher

def run():
    gallery=KnownPersonGallery();gallery.add("PERSON-0003","Husan",[np.array([1,0,0]),np.array([.98,.02,0])]);gallery.add("PERSON-0004","Ali",[np.array([0,1,0])])
    match=KnownPersonMatcher(gallery,.55,.8,.05).match(np.array([.99,.01,0]))
    print("GLOBAL UNK-000007");print("face candidate quality: 0.88");print(f"best match: {match.person_id} {match.name}");print(f"similarity: {match.similarity:.3f}");print(f"second best: {match.second_best_similarity:.3f}");print(f"decision: {match.decision.value}")
if __name__=="__main__":run()
