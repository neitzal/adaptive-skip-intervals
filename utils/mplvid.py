import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np


class BarChartVideoBuilder:
    def __init__(self, img_height, img_width, dpi, style='default',
                 xlim=None, ylim=None,
                 xticks=None, xticklabels=None, yticks=None, yticklabels=None,
                 ):
        with plt.style.context(style):
            self.fig, self.ax = plt.subplots(
                figsize=(img_width / dpi, img_height / dpi), dpi=dpi)

        def maybe_call(method, value):
            if value is not None:
                method(value)

        maybe_call(self.ax.set_xlim, xlim)
        maybe_call(self.ax.set_ylim, ylim)
        maybe_call(self.ax.set_xticks, xticks)
        maybe_call(self.ax.set_xticklabels, xticklabels)
        maybe_call(self.ax.set_yticks, yticks)
        maybe_call(self.ax.set_yticklabels, yticklabels)

        self.fig.tight_layout()

        self.rects = None
        self.img_frames = []

    def add_frame(self, x, heights):
        if self.rects is None:
            self.rects = self.ax.bar(x, heights)
        else:
            for rect, height in zip(self.rects, heights):
                rect.set_height(abs(height))
                rect.set_y(0 if height >= 0 else height)

        self.img_frames.append(self._render())

    def _render(self):
        self.fig.canvas.draw()

        img_array = np.fromstring(self.fig.canvas.tostring_rgb(),
                                  dtype=np.uint8, sep='').reshape(
            self.fig.canvas.get_width_height()[::-1] + (3,))
        return img_array

    def get_rendered_frames(self):
        plt.close(self.fig)
        return np.array(self.img_frames)

    def __del__(self):
        try:
            plt.close(self.fig)
        except AttributeError:
            pass


def get_bar_chart_video(xs, heights_frames,
                        img_height, img_width, dpi, style='default',
                        xlim=None, ylim=None,
                        xticks=None, xticklabels=None, yticks=None, yticklabels=None
                        ):
    builder = BarChartVideoBuilder(img_height, img_width, dpi, style,
                                   xlim, ylim,
                                   xticks, xticklabels, yticks, yticklabels)
    xs = np.asarray(xs)
    heights_frames = np.asarray(heights_frames)

    if heights_frames.ndim != 2:
        raise ValueError('heights_frames has to be a matrix (frames, values)')

    if xs.ndim == 1:
        xs = np.repeat(xs[np.newaxis], heights_frames.shape[0], axis=0)

    for x, heights in zip(xs, heights_frames):
        builder.add_frame(x, heights)

    return builder.get_rendered_frames()


if __name__ == '__main__':
    xs = []
    ys = []

    for t in np.linspace(0, 4 * np.pi, 300):
        xs.append(np.arange(3))
        ys.append([-1, np.sin(t), np.cos(t)])

    frames = get_bar_chart_video(xs, ys, 200, 150, 80,
                                 style='default',
                                 xticks=[1, 2, 3], xticklabels=['$\\alpha$', 'b', 'c'])
    frames = list(frames)

    print('frames[0].shape', frames[0].shape)

    print('frames', frames)

    import moviepy.editor as mpy

    clip = mpy.ImageSequenceClip(frames, fps=60)
    clip.write_videofile('/tmp/mplvidtest.mp4')
