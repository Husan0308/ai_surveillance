with open('backend/ai/ai_worker.py', 'r') as f:
    c = f.read()

# 1. person_recognized: permanent -> 30s cooldown
old1 = """        if etype == "person_recognized" and person_id is not None:
            _cls = type(self)
            if not hasattr(_cls, "_global_rec_logged"):
                _cls._global_rec_logged = set()
            if person_id in _cls._global_rec_logged:
                return  # allaqachon ko'rilgan → qayta yozma
            _cls._global_rec_logged.add(person_id)"""

new1 = """        if etype == "person_recognized" and person_id is not None:
            _cls = type(self)
            if not hasattr(_cls, "_global_rec_ts"):
                _cls._global_rec_ts = {}
            import time as _t
            _now = _t.time()
            if person_id in _cls._global_rec_ts and _now - _cls._global_rec_ts[person_id] < 30:
                return  # 30s ichida qayta yozma
            _cls._global_rec_ts[person_id] = _now"""

# 2. unknown_detected: permanent -> 30s cooldown
old2 = """        if etype == "unknown_detected":
            _cls2 = type(self)
            if not hasattr(_cls2, "_global_unknown_logged"):
                _cls2._global_unknown_logged = set()
            if name in _cls2._global_unknown_logged:
                return  # allaqachon ko'rilgan → qayta yozma
            _cls2._global_unknown_logged.add(name)"""

new2 = """        if etype == "unknown_detected":
            _cls2 = type(self)
            if not hasattr(_cls2, "_global_unknown_ts"):
                _cls2._global_unknown_ts = {}
            import time as _t
            _now = _t.time()
            if name in _cls2._global_unknown_ts and _now - _cls2._global_unknown_ts[name] < 30:
                return  # 30s ichida qayta yozma
            _cls2._global_unknown_ts[name] = _now"""

changed = False
if old1 in c:
    c = c.replace(old1, new1)
    print("✅ person_recognized: permanent -> 30s cooldown")
    changed = True
else:
    print("⚠ person_recognized pattern topilmadi")

if old2 in c:
    c = c.replace(old2, new2)
    print("✅ unknown_detected: permanent -> 30s cooldown")
    changed = True
else:
    print("⚠ unknown_detected pattern topilmadi")

if changed:
    with open('backend/ai/ai_worker.py', 'w') as f:
        f.write(c)
    print("💾 ai_worker.py saqlandi")
else:
    print("❌ Hech narsa o'zgarmadi")
