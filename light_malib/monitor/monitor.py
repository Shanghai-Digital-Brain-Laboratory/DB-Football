# MIT License

# Copyright (c) 2022 DigitalBrain, Yan Song and He jiang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from light_malib.utils.logger import Logger
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt


class Monitor:
    """
    TODO(jh): wandb etc
    TODO(jh): more functionality.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.writer = SummaryWriter(log_dir=cfg.expr_log_dir)

    def get_expr_log_dir(self):
        return self.cfg.expr_log_dir

    def add_scalar(self, tag, scalar_value, global_step, *args, **kwargs):
        self.writer.add_scalar(tag, scalar_value, global_step, *args, **kwargs)

    def add_multiple_scalars(
        self, main_tag, tag_scalar_dict, global_step, *args, **kwargs
    ):
        for tag, scalar_value in tag_scalar_dict.items():
            tag = main_tag + tag
            self.writer.add_scalar(tag, scalar_value, global_step, *args, **kwargs)

    def add_scalars(self, main_tag, tag_scalar_dict, global_step, *args, **kwargs):
        self.writer.add_scalars(main_tag, tag_scalar_dict, global_step, *args, **kwargs)

    def add_array(
        self, main_tag, image_array, xpid, ypid, global_step, color, *args, **kwargs
    ):
        array_to_rgb(
            self.writer, main_tag, image_array, xpid, ypid, global_step, color, **kwargs
        )

    def close(self):
        self.writer.close()


def array_to_rgb(writer, tag, array, xpid, ypid, steps, color="bwr", **kwargs):
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

    writer.add_image(tag, img, steps)
    plt.close(fig)
