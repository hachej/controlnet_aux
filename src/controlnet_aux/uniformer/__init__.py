import os
import numpy as np
import os
from huggingface_hub import hf_hub_download

from mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
import pathlib
from ..open_pose.util import HWC3, resize_image

class UniformerDetector:
    def __init__(self, model):
        self.model = model.eval()

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path, filename=None, cache_dir=None):
        filename = filename or "annotator/ckpts/upernet_global_small.pth"
        model_path = hf_hub_download(pretrained_model_or_path, filename, cache_dir=cache_dir)
        
        base_path = pathlib.Path(__file__).parent.resolve()

        config_file = os.path.join(base_path , "..", "..","exp", "upernet_global_small", "config.py")
        model = init_segmentor(config_file, model_path).cuda()
        return cls(model)

    def __call__(self, input_image, detect_resolution=512):
        
        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)
        input_image = HWC3(input_image)
        input_image = resize_image(input_image, detect_resolution)

        label_image = inference_segmentor(self.model, input_image)
        res_img = show_result_pyplot(self.model, input_image, label_image, get_palette('ade'), opacity=1)
    
        return res_img, label_image

