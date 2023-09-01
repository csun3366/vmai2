import os
#os.chdir('/content/so-vits-svc') # force working-directory to so-vits-svc - this line is just for safety and is probably not required

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

import huggingface_hub
import os
import shutil

class HFModels:
    def __init__(self, repo = "therealvul/so-vits-svc-4.0", 
            model_dir = "hf_vul_models"):
        self.model_repo = huggingface_hub.Repository(local_dir=model_dir,
            clone_from=repo, skip_lfs_files=True)
        self.repo = repo
        self.model_dir = model_dir

        self.model_folders = sorted(os.listdir(model_dir))
        print(self.model_folders)
        self.model_folders.remove('.git')
        self.model_folders.remove('.gitattributes')

    def list_models(self):
        return self.model_folders

    # Downloads model;
    # copies config to target_dir and moves model to target_dir
    def download_model(self, model_name, target_dir):
        if not model_name in self.model_folders:
            raise Exception(model_name + " not found")
        model_dir = self.model_dir
        charpath = os.path.join(model_dir,model_name)

        gen_pt = next(x for x in os.listdir(charpath) if x.startswith("G_"))
        cfg = next(x for x in os.listdir(charpath) if x.endswith("json"))
        try:
          clust = next(x for x in os.listdir(charpath) if x.endswith("pt"))
        except StopIteration as e:
          print("Note - no cluster model for "+model_name)
          clust = None

        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)

        gen_dir = huggingface_hub.hf_hub_download(repo_id = self.repo,
            filename = model_name + "/" + gen_pt) # this is a symlink
        
        if clust is not None:
          clust_dir = huggingface_hub.hf_hub_download(repo_id = self.repo,
              filename = model_name + "/" + clust) # this is a symlink
          shutil.move(os.path.realpath(clust_dir), os.path.join(target_dir, clust))
          clust_out = os.path.join(target_dir, clust)
        else:
          clust_out = None

        shutil.copy(os.path.join(charpath,cfg),os.path.join(target_dir, cfg))
        shutil.move(os.path.realpath(gen_dir), os.path.join(target_dir, gen_pt))

        return {"config_path": os.path.join(target_dir,cfg),
            "generator_path": os.path.join(target_dir,gen_pt),
            "cluster_path": clust_out}

# Example usage
# vul_models = HFModels()
# print(vul_models.list_models())
# print("Applejack (singing)" in vul_models.list_models())
# vul_models.download_model("Applejack (singing)","models/Applejack (singing)")
downloader = Downloader2()
print("Finished!")
