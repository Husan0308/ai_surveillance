from __future__ import annotations

import sys

from . import rfdetr_detection as runtime

# Keep the runtime module focused on detector logic while this entrypoint owns
# process-level stderr/exit handling.
runtime.sys = sys


def main() -> int:
    return runtime.main()


if __name__ == "__main__":
    raise SystemExit(main())
