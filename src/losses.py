import tensorflow as tf

def get_ssim_loss(num_frames):

    def ssim_loss(y_true, y_pred):

        reconstruction_ssim_loss = 0
        for frame in range(num_frames):
            reconstruction_ssim_loss += (1 - tf.reduce_mean(tf.image.ssim(tf.image.rgb_to_yuv(y_true[:, frame]), tf.image.rgb_to_yuv(y_pred[:, frame]), max_val=1.0)))

        return reconstruction_ssim_loss

    return ssim_loss

def get_palette_loss(num_frames):

    #Not implemented, leaving in because my brain is goo

    def palette_loss(y_true, y_pred):
        '''
        Defines the loss per pixel as the distance to the closest color in the palette, as defined by the colors in the first and last frames of y_true
        :param y_true:
        :param y_pred:
        :return:
        '''

        first_frame_colors = tf.raw_ops.UniqueV2(y_true[:, 0], axis=(1, 2))
        last_frame_colors = tf.raw_ops.UniqueV2(y_true[:, -1], axis=(1, 2))

        all_unique_colors = tf.raw_ops.UniqueV2(tf.concat((first_frame_colors, last_frame_colors))) # batch x N x 3 matrix

        # Loss = min(tf.sum(squared_dif(pixel color, all_unique_colors)))

        y_pred_colors = tf.raw_ops.UniqueV2(y_pred, axis=(2, 3)) # batch x M x 3 matrix

        all_unique_colors = tf.expand_dims(all_unique_colors, 1) # batch x 1 x N x 3
        y_pred_colors = tf.expand_dims(y_pred_colors, 2) # batch x M x 1 x 3

        squared_error = tf.math.squared_difference(all_unique_colors, y_pred_colors)

        min_color_error = tf.math.reduce_min(squared_error, axis=1)

        loss = tf.math.reduce_mean()


        print("lets GET this BREAD homes homes homes")

        pass