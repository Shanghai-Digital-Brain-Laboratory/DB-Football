# Copyright 2022 Digital Brain Laboratory, Yan Song and He jiang
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from light_malib.utils.logger import Logger
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb

class Monitor:
    """
    TODO(jh): wandb etc
    TODO(jh): more functionality.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.monitor_type = cfg.monitor.get('type', 'local')
        # if self.monitor_type == 'local':
        self.writer = SummaryWriter(log_dir=cfg.expr_log_dir)

        if self.monitor_type == 'remote':
            wandb.init(
                project=f'{cfg.expr_group}-{cfg.expr_name}',
                config = cfg
            )



    def get_expr_log_dir(self):
        return self.cfg.expr_log_dir

    def add_scalar(self, tag, scalar_value, global_step, *args, **kwargs):
        if self.monitor_type == 'local':
            self.writer.add_scalar(tag, scalar_value, global_step, *args, **kwargs)
        elif self.monitor_type == 'remote':
            wandb.log({tag: scalar_value,
                       })


    def add_multiple_scalars(
        self, main_tag, tag_scalar_dict, global_step, *args, **kwargs
    ):
        for tag, scalar_value in tag_scalar_dict.items():
            tag = main_tag + tag
            if self.monitor_type == 'local':
                self.writer.add_scalar(tag, scalar_value, global_step, *args, **kwargs)
            elif self.monitor_type == 'remote':
                wandb.log({tag: scalar_value})



    def add_scalars(self, main_tag, tag_scalar_dict, global_step, *args, **kwargs):
        if self.monitor_type == 'local':
            self.writer.add_scalars(main_tag, tag_scalar_dict, global_step, *args, **kwargs)
        elif self.monitor_type == 'remote':
            log_dict = {}
            for tag, scalar in tag_scalar_dict.items():
                log_dict[f'{main_tag}_{tag}'] = scalar
            wandb.log(log_dict)


    def add_array(
        self, main_tag, image_array, xpid, ypid, global_step, color, *args, **kwargs
    ):
        array_to_rgb(
            self.writer, main_tag, image_array, xpid, ypid, global_step, color, self.monitor_type,**kwargs
        )

    def close(self):
        self.writer.close()
        wandb.finish()


def array_to_rgb(writer, tag, array, xpid, ypid, steps, color="bwr", mode='local',**kwargs):
    matrix = np.array(array)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    color_map = plt.cm.get_cmap(color)
    cax = ax.matshow(matrix, cmap=color_map)
    ax.set_xticklabels([""] + xpid, rotation=90)
    ax.set_yticklabels([""] + ypid)

    if kwargs.get("show_text", False):
        for (j, i), label in np.ndenumerate(array):
            ax.text(i, j, f"{label:.1f}", ha="center", va="center")

    fig.colorbar(cax)
    ax.grid(False)
    plt.tight_layout()

    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = img / 255.0
    img = img.transpose(2, 0, 1)

    if mode == 'local':
        writer.add_image(tag, img, steps)
    elif mode == 'remote':
        wandb.log({tag: plt})
    else:
        Logger.warning("monitor mode is not implemented")
        pass

    plt.close(fig)
