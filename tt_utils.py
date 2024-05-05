import os


def create_output_directory(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

