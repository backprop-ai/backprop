from PIL import Image
from io import BytesIO
from typing import Union, List
import base64
import os

def base64_to_img(image: Union[str, List[str]]):
    """
    Returns PIL Image objects of base64 encoded images
    """
    is_list = False

    if type(image) == list:
        is_list = True

    if not is_list:
        image = [image]
    
    images = []
    for img in image:
        if not isinstance(img, Image.Image):
            # Not bytes
            if type(img) == str:
                img = img.split(",")[-1]

            img = BytesIO(base64.b64decode(img))
            img = Image.open(img)
        images.append(img)
    
    if not is_list:
        images = images[0]

    return images

def path_to_img(image: Union[str, List[str]]):
    """
    Returns PIL Image objects of paths to images
    """
    is_list = False

    if type(image) == list:
        is_list = True

    if not is_list:
        image = [image]
    
    images = []
    for img in image:
        if not isinstance(img, Image.Image):
            img = Image.open(img)
        
        images.append(img)

    if not is_list:
        images = images[0]
    
    return images

def img_to_base64(image: Union[Image.Image, List[Image.Image]]):
    """
    Returns base64 encoded strings of PIL Image objects
    """
    is_list = False

    if type(image) == list:
        is_list = True

    if not is_list:
        image = [image]

    images = []
    for img in image:
        buffered = BytesIO()
        img.save(buffered, format=img.format)
        img = base64.b64encode(buffered.getvalue())
        images.append(img)

    if not is_list:
        images = images[0]
    
    return images