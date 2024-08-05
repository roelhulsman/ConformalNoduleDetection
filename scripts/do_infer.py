# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import json
from typing import Dict

from monai.transforms import Invertd, SaveImaged

import monailabel
from monailabel.interfaces.app import MONAILabelApp
from monailabel.interfaces.tasks.infer_v2 import InferTask
from monailabel.interfaces.tasks.scoring import ScoringMethod
from monailabel.interfaces.tasks.strategy import Strategy
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.tasks.activelearning.first import First
from monailabel.tasks.activelearning.random import Random
from monailabel.tasks.infer.bundle import BundleInferTask
from monailabel.tasks.scoring.epistemic_v2 import EpistemicScoring
from monailabel.tasks.train.bundle import BundleTrainTask
from monailabel.utils.others.generic import get_bundle_models, strtobool

logger = logging.getLogger(__name__)


class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies, conf):
        self.models = get_bundle_models(app_dir, conf)
        # Add Epistemic model for scoring
        self.epistemic_models = (
            get_bundle_models(app_dir, conf, conf_key="epistemic_model") if conf.get("epistemic_model") else None
        )
        if self.epistemic_models:
            # Get epistemic parameters
            self.epistemic_max_samples = int(conf.get("epistemic_max_samples", "0"))
            self.epistemic_simulation_size = int(conf.get("epistemic_simulation_size", "5"))
            self.epistemic_dropout = float(conf.get("epistemic_dropout", "0.2"))

        super().__init__(
            app_dir=app_dir,
            studies=studies,
            conf=conf,
            name=f"MONAILabel - Zoo/Bundle ({monailabel.__version__})",
            description="DeepLearning models provided via MONAI Zoo/Bundle",
            version=monailabel.__version__,
        )

    def init_infers(self) -> Dict[str, InferTask]:
        infers: Dict[str, InferTask] = {}
        #################################################
        # Models
        #################################################

        for n, b in self.models.items():
            if "deepedit" in n:
                # Adding automatic inferer
                i = BundleInferTask(b, self.conf, type="segmentation")
                logger.info(f"+++ Adding Inferer:: {n}_seg => {i}")
                infers[n + "_seg"] = i
                # Adding inferer for managing clicks
                i = BundleInferTask(b, self.conf, type="deepedit")
                logger.info("+++ Adding DeepEdit Inferer")
                infers[n] = i
            else:
                i = BundleInferTask(b, self.conf)
                logger.info(f"+++ Adding Inferer:: {n} => {i}")
                infers[n] = i

        return infers


"""
Example to run infer task locally without actually running MONAI Label Server
"""


def main():
    import argparse
    import shutil
    from pathlib import Path

    from monailabel.utils.others.generic import device_list, file_ext

    os.putenv("MASTER_ADDR", "127.0.0.1")
    os.putenv("MASTER_PORT", "1234")

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(process)s] [%(threadName)s] [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    home = str(Path.home())
    studies = f"{home}/Datasets/Radiology"

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--studies", default=studies)
    parser.add_argument("--seg_ct", action="store_true")
    parser.add_argument("--det_nd", action="store_true")
    parser.add_argument("-t", "--test", default="infer", choices=("train", "infer", "batch_infer", "download"))
    parser.add_argument("--app_dir", default="apps/monaibundle/")
    args = parser.parse_args()

    studies = args.studies

    models = []
    if args.seg_ct: 
        models.append("wholeBody_ct_segmentation")
        args.model = "wholeBody_ct_segmentation"
    if args.det_nd: 
        models.append("lung_nodule_ct_detection")
        args.model = "lung_nodule_ct_detection"

    conf = {
        "models": ",".join(models),
        "preload": "false",
    }

    app = MyApp(app_dir, studies, conf)

    # Infer
    if args.test == "infer":
        sample = app.next_sample(request={"strategy": "first"})
        image_id = sample["id"]
        image_path = sample["path"]

        # Run on all devices
        for device in device_list():
            res = app.infer(request={"model": args.model, "image": image_id, "device": device})
            label = res["file"]
            label_json = res["params"]
            test_dir = os.path.join(args.studies, "test_labels")
            os.makedirs(test_dir, exist_ok=True)

            label_file_ext = ""
            
            if args.seg_ct:
                label_file_ext = "_seg_ct" + file_ext(image_path)
                label_json_path = os.path.join(test_dir, "labels_seg_ct.json")
                
            if args.det_nd:
                label_file_ext = "_det_nd" + ".json"
                label_json_path = os.path.join(test_dir, "labels_det_nd.json")
                               
            label_file = os.path.join(test_dir, image_id + label_file_ext)
            shutil.move(label, label_file)
            

            with open(label_json_path, "w") as fp:
                json.dump(label_json, fp, sort_keys=True, indent=2)

            
            #print(label_json)
            print(f"++++ Image File: {image_path}")
            print(f"++++ Label File: {label_file}")
            print(f"++++ Label File: {label_json_path}")
            break
        return


if __name__ == "__main__":
    main()
