from zipfile import ZipFile
from pathlib import Path
import tarfile
import gdown
import urllib
import os
import subprocess
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)
        
class Downloader2:
    LINUX_MEGATOOLS_URL = ("https://megatools.megous.com/builds/builds/"
        "megatools-1.11.1.20230212-linux-x86_64.tar.gz")
    def __init__(self):
        self.mega_setup()
        pass

    def mega_setup(self):
        MEGATOOLS_TAR_PATH = Downloader2.LINUX_MEGATOOLS_URL.split("/")[-1]
        MEGATOOLS_FOLDER_PATH = (
            Downloader2.LINUX_MEGATOOLS_URL.split("/")[-1].removesuffix(
                ".tar.gz"))
        self.abs_megatools_path = os.path.abspath(MEGATOOLS_FOLDER_PATH)

        urllib.request.urlretrieve(
            url=Downloader2.LINUX_MEGATOOLS_URL,
            filename=MEGATOOLS_TAR_PATH)
        with tarfile.open(MEGATOOLS_TAR_PATH) as tar:
            tar.extractall()
            tar.close()
        os.unlink(MEGATOOLS_TAR_PATH)
        assert os.path.exists(self.abs_megatools_path)

        pass

    def megadown(self, url, filename):
        cmd = (os.path.join(self.abs_megatools_path,'megatools')+
            " dl "+"--print-names "+(
            "--path "+filename+" " if filename else "")+url)
        proc = subprocess.run(cmd, shell=True)
        if proc.returncode != 0:
            raise Exception('megadown failed -- cmd: '+cmd)
        return filename

    def request_url_with_progress_bar(self, url, filename):
        def download_url(url, filename):
            with DownloadProgressBar(unit='B', unit_scale=True,
                miniters=1, desc=url.split('/')[-1]) as t:
                return urllib.request.urlretrieve(
                    url, filename=filename, reporthook=t.update_to)
        return download_url(url, filename)

    def download(self, url, filename):
        if "drive.google.com" in url:
            print("Downloading Google Drive file "+url)
            return gdown.download(url, filename, quiet=False, fuzzy=True)
        elif "mega.nz" in url:
            print("Downloading MEGA file "+url)
            # There is no other way to determine the file name
            # from megatools prior to downloading without authentication
            # so we set it to a placeholder
            import uuid
            return self.megadown(url, filename=str(uuid.uuid4())+".zip")
        else:
            print("Downloading direct "+url)
            local_filename, headers = (
                self.request_url_with_progress_bar(url, filename))
            return local_filename


import fnmatch
def default_next(x):
    try:
        return next(x)
    except StopIteration:
        return None
    
def zip_extract(zipfile, model_dir):
    model_folder_name = Path(zipfile).stem
    model_folder_path = os.path.join(model_dir,
        Path(zipfile).stem)
    with ZipFile(zipfile, 'r') as f:
        member_infos = f.infolist()
        generator = default_next(
            x for x in member_infos if fnmatch.fnmatch(x.filename, '*G_*.pth'))
        config_json = default_next(
            x for x in member_infos if fnmatch.fnmatch(x.filename, '*.json'))
        cluster_pt = default_next(
            x for x in member_infos if fnmatch.fnmatch(x.filename, '*.pt'))

        generator.filename = generator.filename.split('/')[-1]
        config_json.filename = config_json.filename.split('/')[-1]

        if (not generator):
            print("Could not find G_*.pth in "+zipfile)
            return
        if (not config_json):
            print("Could not find config.json in "+zipfile)
            return
        f.extract(generator, path=model_folder_path)
        f.extract(config_json, path=model_folder_path)

        if cluster_pt:
            cluster_pt.filename = cluster_pt.filename.split('/')[-1]
            f.extract(cluster_pt, path=model_folder_path)
    print("Cleaning "+zipfile)
    os.remove(zipfile)

downloader = Downloader2()
downloader.download("https://huggingface.co/therealvul/so-vits-svc-4.0-init/resolve/main/checkpoint_best_legacy_500.pt", filename="hubert/checkpoint_best_legacy_500.pt")
