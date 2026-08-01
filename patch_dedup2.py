with open('backend/ai/ai_worker.py', 'r') as f:
    c = f.read()

# 1. Global permanent dedup ni O'CHIRISH (person_recognized)
old1 = """        # ✅ KUCHLI DEDUP: person_recognized bir person uchun 10 min ichida faqat 1 marta
        if etype == "person_recognized" and person_id is not None:
            _cls = type(self)
            if not hasattr(_cls, "_global_rec_logged"):
                _cls._global_rec_logged = set()
            if person_id in _cls._global_rec_logged:
                return  # allaqachon ko'rilgan → qayta yozma
            _cls._global_rec_logged.add(person_id)"""

new1 = """        # ✅ Global permanent dedup OLIB TASHLANDI"""

# 2. Global permanent dedup ni O'CHIRISH (unknown_detected)
old2 = """        # ✅ GLOBAL DEDUP: unknown_detected — ko'rdimi boldi, qayta yozmaydi (permanent)
        if etype == "unknown_detected":
            _cls2 = type(self)
            if not hasattr(_cls2, "_global_unknown_logged"):
                _cls2._global_unknown_logged = set()
            if name in _cls2._global_unknown_logged:
                return  # allaqachon ko'rilgan → qayta yozma
            _cls2._global_unknown_logged.add(name)"""

new2 = """        # ✅ Global permanent dedup OLIB TASHLANDI"""

# 3. key ni TRACK_ID ga asoslangan qilish (person_id emas)
old3 = """        if person_id is not None:
            key = (self.camera_id, person_id)  # kamera + person (track_id emas!)
        else:
            key = (self.camera_id, tr.id)  # kamera + track_id (unknown uchun)"""

new3 = """        key = (self.camera_id, tr.id)  # kamera + track: bitta track=bitta log, boshqa kamera=yangi log"""

changed = False
for old, new, label in [(old1, new1, "global_rec"), (old2, new2, "global_unknown"), (old3, new3, "key")]:
    if old in c:
        c = c.replace(old, new)
        print(f"✅ {label} o'zgartirildi")
        changed = True
    else:
        print(f"⚠ {label} topilmadi")

if changed:
    with open('backend/ai/ai_worker.py', 'w') as f:
        f.write(c)
    print("💾 ai_worker.py saqlandi")
else:
    print("❌ Hech narsa o'zgarmadi")
