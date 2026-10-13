"""Keep test imports isolated from deployment bindings in the local .env."""

import os


# Several configuration modules are intentionally resolved at import time.
# Earlier test imports may load the operator's .env, so pin the entire pytest
# process to loopback before collection. Tests for remote binding explicitly
# override these values and verify the bearer-token requirement.
os.environ["API_HOST"] = "127.0.0.1"
os.environ["ML_HOST"] = "127.0.0.1"
os.environ.pop("SURVEILLANCE_API_TOKEN", None)
