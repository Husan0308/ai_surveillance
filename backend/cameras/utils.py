from urllib.parse import quote


def is_int_source(source) -> bool:
    if isinstance(source, int):
        return True
    if isinstance(source, str) and source.strip().isdigit():
        return True
    return False


def build_source_url(source, username=None, password=None):
    """
    RTSP URL ni xavfsiz yasaydi.

    Agar parolda @ bo'lsa:
        aps@2026 -> aps%402026

    Agar source ichida allaqachon credentials bo'lsa,
    uni o'zgartirmaydi.
    """

    if is_int_source(source):
        return int(str(source).strip())

    src = str(source).strip()

    if not src:
        return src

    if not username and not password:
        return src

    if "://" not in src:
        return src

    scheme, rest = src.split("://", 1)

    parts = rest.split("/", 1)
    netloc = parts[0]
    path = parts[1] if len(parts) > 1 else ""

    # Agar netloc ichida @ bo'lsa, credentials allaqachon bor deb hisoblaymiz.
    if "@" in netloc:
        return src

    user_enc = quote(str(username or ""), safe="")
    pass_enc = quote(str(password or ""), safe="")

    if user_enc and pass_enc:
        auth = f"{user_enc}:{pass_enc}@"
    elif user_enc:
        auth = f"{user_enc}@"
    else:
        auth = ""

    if path:
        return f"{scheme}://{auth}{netloc}/{path}"

    return f"{scheme}://{auth}{netloc}"