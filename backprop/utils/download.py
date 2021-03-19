import os
import urllib
from tqdm import tqdm

def download(url: str, folder: str, root: str = os.path.expanduser("~/.cache/backprop"), force: bool = False):
    path = os.path.join(root, folder)
    os.makedirs(path, exist_ok=True)
    filename = os.path.basename(url)
    
    download_target = os.path.join(path, filename)

    if os.path.isfile(download_target) and not force:
        return download_target

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length"))) as loop:        
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    return download_target