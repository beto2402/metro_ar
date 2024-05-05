import os
import shutil
from tt_constants import OGS_PATH, RENAMEDS_PATH
from tt_utils import create_output_directory


def rename_files():
    create_output_directory(RENAMEDS_PATH)

    og_rads = sorted(os.listdir(OGS_PATH))

    for i, og_rad_path in enumerate(og_rads):
        new_rad_name = f"rad_{i}.png"

        shutil.copy(f"{OGS_PATH}/{og_rad_path}", f"{RENAMEDS_PATH}/{new_rad_name}")
        

rename_files()