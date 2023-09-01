import os
import downloader
#os.chdir('/content/so-vits-svc') # force working-directory to so-vits-svc - this line is just for safety and is probably not required
downloader.download("https://huggingface.co/therealvul/so-vits-svc-4.0-init/resolve/main/checkpoint_best_legacy_500.pt", filename="hubert/checkpoint_best_legacy_500.pt")
