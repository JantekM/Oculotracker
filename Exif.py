import json
from exif import Image


def save_exif(fname: str, metadata: dict):
    with open(fname, 'rb') as new_image_file:
        img = Image(new_image_file)

    img.user_comment = json.dumps(metadata)
    with open(fname, 'wb') as new_image_file:
        new_image_file.write(img.get_file())


def load_exif(fname: str) -> dict:
    with open(fname, 'rb') as new_image_file:
        img = Image(new_image_file)
    return json.loads(img.user_comment)
