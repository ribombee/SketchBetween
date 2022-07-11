import tensorflow as tf
import numpy as np
import cv2 as cv
from pathlib import Path


def _sketchify(image):
    # Blur
    kernel_size = np.random.randint(3, 9)  # Random due to the existence of differently scaled pixel-art
    blurred_image = cv.blur(image, (kernel_size, kernel_size))

    canny_edges = cv.Canny(blurred_image, 50, 150)
    canny_rgb = cv.cvtColor(canny_edges, cv.COLOR_GRAY2RGB)  # This is white-on-black
    canny_inverted = 255 - canny_rgb  # This is black on white
    return canny_inverted


def _sketchify_mixed(image):
    # Blur
    kernel_sizes = range(3,9,2)

    cannies = []
    for kernel_size in kernel_sizes:
        blurred_image = cv.blur(image, (kernel_size, kernel_size))

        canny_edges = cv.Canny(blurred_image, 50, 150)
        cannies.append(canny_edges)

    canny_einsum = np.einsum('ijk, ijk->jk', np.array(cannies),
                             np.array(cannies)) / 3  # Dividing by 3 gives us numbers in the range [0,1]
    canny_rgb = cv.cvtColor((canny_einsum * 255).astype(np.uint16), cv.COLOR_GRAY2RGB)  # This is white-on-black
    canny_inverted = 255 - canny_rgb  # This is black on white

    return canny_inverted


class VQVAEAnimatedGenerator(tf.keras.utils.Sequence):

    def __init__(self, dataloc: Path, batch_size: int, shuffle=True, animation_len = 6, augment = False, do_sketches=True, include_final_keyframe=True):
        self.dataloc = dataloc
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.animation_len = animation_len
        self.augment = augment
        self.do_sketches = do_sketches
        self.include_final_keyframe = include_final_keyframe

        self.animation_locs = self.separate_animations(self.dataloc)
        self.idxs = self.create_indices()


    def create_indices(self):

        unusable = 0
        idxs = []

        for animation_index, animation in enumerate(self.animation_locs):

            if len(animation) < self.animation_len:
                # We cannot use this animation.
                unusable += 1
                continue

            for frame_index, starting_frame in enumerate(animation[:-(self.animation_len -1)]):
                idxs.append((animation_index, frame_index))

        print(f"FOUND {unusable} UNUSABLE ANIMATIONS")
        return idxs


    def _augment(self, frames):

        flip_horizontal = np.random.rand() > 0.5
        shift_hue = np.random.rand() > 0.5
        shift_sat = np.random.rand() > 0.5
        pixellate = np.random.rand() > 0.9


        hue_shift_amount =  np.random.randint(low=0, high=180)
        sat_mult = 1 + ( 0.5 - np.random.rand()) / 5 # Random between 0.8 and 1.2

        augmented_frames = []
        for image in frames:

            if flip_horizontal:
                image = cv.flip(image, 1)

            hsv_image = cv.cvtColor(image, cv.COLOR_RGB2HSV)
            hue, sat, val = cv.split(hsv_image)

            if shift_hue:
                hue = (hue + hue_shift_amount) % 180

            if shift_sat:
                sat = (sat * sat_mult)
                sat = np.clip(sat, 0, 255)
                sat = sat.astype('uint8')

            if pixellate:
                w, h = (32, 32)

                temp = cv.resize(image, (w, h), interpolation=cv.INTER_LINEAR)
                image = cv.resize(temp, (128, 128), interpolation= cv.INTER_NEAREST)

            hsv_image = cv.merge((hue, sat, val))
            image = cv.cvtColor(hsv_image, cv.COLOR_HSV2RGB)

            augmented_frames.append(image)

        return augmented_frames


    def _shuffle_indices(self):
        np.random.shuffle(self.idxs)  # Shuffles first axis

    def _data_generation(self, start, stop):

        X = []
        Y = []

        for index in self.idxs[start:stop]:
            frames = self.load_animation(self.animation_locs[index[0]][index[1]: index[1] + self.animation_len])

            if self.augment:
                frames = self._augment(frames)

            sketchy_frames = frames.copy()
            if self.do_sketches:
                if self.include_final_keyframe:
                    sketchy_frames[1:-1] = [_sketchify_mixed(frame) for frame in sketchy_frames[1:-1]]
                else:
                    sketchy_frames[1:] = [_sketchify_mixed(frame) for frame in sketchy_frames[1:]]
            else:
                sketchy_frames[1:-1] = np.zeros_like(sketchy_frames[1:-1])

            # Normalization
            frames = [frame / 255 for frame in frames]
            sketchy_frames = [frame / 255 for frame in sketchy_frames]

            X.append(np.stack(sketchy_frames, axis=0))
            Y.append(np.stack(frames, axis=0))

        return np.array(X), np.array(Y)

    def __len__(self):
        return len(self.idxs) // self.batch_size

    def __getitem__(self, index):

        start = index*self.batch_size
        stop = (index +1)*self.batch_size

        X, Y = self._data_generation(start, stop)

        return X, Y

    def on_epoch_end(self):

        if self.shuffle:
            self._shuffle_indices()

    def load_animation(self, frames_loc):

        images = []
        for frame_loc in frames_loc:
            image = cv.imread(str(frame_loc))
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            images.append(image)

        return images

    def separate_animations(self, data_loc: Path):

        filenames = list(data_loc.glob('*.png'))
        filenames_no_frame = set(['_'.join(str(filename).split('_')[:-1]) for filename in filenames]) # Remove the '_frame#.png" from the end.
        all_frames_locs = []

        for filename in filenames_no_frame:

            filename_stem = Path(filename).name
            frame_locs = data_loc.glob(filename_stem + "_*.png")
            frame_locs = sorted(frame_locs, key=lambda x:int(str(x.name).split("_frame")[-1][:-4]))
            all_frames_locs.append(list(frame_locs))

        return all_frames_locs
