"""
Python module serving as a project/extension template.
"""

import os


# Most entry points only need robots/utils. Let callers opt out of task
# auto-registration when they do not want Gym config imports as a side effect.
if os.environ.get("WHOLE_BODY_TRACKING_IMPORT_TASKS", "1") == "1":
    from .tasks import *  # noqa: F401,F403
