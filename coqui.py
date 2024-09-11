from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import time

# Load the configuration
config = XttsConfig()
config.load_json("./XTTS-v2/config.json")

# Initialize the model and load the checkpoint
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="./XTTS-v2/")
model.cuda()
