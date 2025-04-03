
## MY implementation for trivial augment wide with tensorflow

import tensorflow as tf
class TrivialAugmentWide(tf.keras.layers.Layer):
    def __init__(self, num_magnitude_bins=31, interpolation='nearest', fill=None, exclude_ops=None, **kwargs):
        super().__init__(**kwargs)
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill
        self.exclude_ops = exclude_ops if exclude_ops else []
        self.ops_space = self._augmentation_space(num_magnitude_bins)
        self.AFFINE_TRANSFORM_INTERPOLATIONS = ("nearest","bilinear")
        self.AFFINE_TRANSFORM_FILL_MODES = ("constant","nearest","wrap","reflect")

        # Filter out excluded operations and create TensorFlow-compatible structures
        self.op_names = tf.constant([name for name in self.ops_space.keys() if name not in self.exclude_ops])
        
        # Ensure all magnitudes are cast to float32 and padded to match the shape [num_magnitude_bins]
        def pad_magnitude(magnitude):
            if magnitude.shape.rank == 0:  # Scalar tensor
                return tf.fill([self.num_magnitude_bins], magnitude)
            return magnitude

        self.op_magnitudes = tf.stack([
            pad_magnitude(tf.cast(params[0], tf.float32))
            for name, params in self.ops_space.items() if name not in self.exclude_ops
        ])
        
        # Signed flags remain as a boolean tensor
        self.op_signed = tf.constant([params[1] for name, params in self.ops_space.items() if name not in self.exclude_ops])

    def _augmentation_space(self, num_bins):
        return {
            "Identity": (tf.constant(0.0), False),
            "ShearX": (tf.linspace(0.0, 0.99, num_bins), True),
            "ShearY": (tf.linspace(0.0, 0.99, num_bins), True),
            "TranslateX": (tf.linspace(0.0, 32.0, num_bins), True),
            "TranslateY": (tf.linspace(0.0, 32.0, num_bins), True),
            "Rotate": (tf.linspace(0.0, 135.0, num_bins), True),
            "Brightness": (tf.linspace(0.0, 0.99, num_bins), True),
            "Color": (tf.linspace(0.0, 0.99, num_bins), True),
            "Contrast": (tf.linspace(0.0, 0.99, num_bins), True),
            "Sharpness": (tf.linspace(0.0, 0.99, num_bins), True),
            "Posterize": ( tf.cast(8 - tf.round(tf.cast(tf.range(num_bins), tf.float32) / ((num_bins - 1) / 6)), tf.uint8), False, ),
            "Solarize": (tf.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (tf.constant(0.0), False),
            "Equalize": (tf.constant(0.0), False),
        }

    def _augment_one(self, img):
        # Ensure input is in [0, 255] range and uint8 type
        input_image_type = img.dtype
        if img.dtype != tf.uint8:
            img = tf.cast(tf.clip_by_value(img, 0, 255), tf.uint8)
        
        # Use a default fill value if self.fill is None
        fill = self.fill if self.fill is not None else 0.0
        
        # Randomly select an operation
        op_index = tf.random.uniform(shape=(), maxval=tf.shape(self.op_names)[0], dtype=tf.int32)
        op_name = tf.gather(self.op_names, op_index)
        magnitudes = tf.gather(self.op_magnitudes, op_index)
        signed = tf.gather(self.op_signed, op_index)
        
        # Randomly select a magnitude using tf.cond to handle symbolic rank
        magnitude = tf.cond(
            tf.greater(tf.rank(magnitudes), 0),
            lambda: tf.gather(magnitudes, tf.random.uniform(shape=(), maxval=tf.shape(magnitudes)[0], dtype=tf.int32)),
            lambda: tf.constant(0.0, dtype=tf.float32)
        )
        
        # Apply random sign if required
        magnitude = tf.cond(
            signed,
            lambda: magnitude * tf.cond(tf.random.uniform(shape=()) > 0.5, lambda: 1.0, lambda: -1.0),
            lambda: magnitude
        )
        
        # Apply the chosen operation
        img = self._apply_op(img, op_name, magnitude, fill=fill)
        
        # Final clip and cast back to uint8
        return tf.cast(img, dtype=input_image_type)

    @tf.function
    def call(self, img):
        # Check if the rank is statically known
        if img.shape.rank is not None:
            if img.shape.rank == 4:
                return tf.map_fn(self._augment_one, img, fn_output_signature=tf.uint8)
            else:
                return self._augment_one(img)
        else:
            # When rank is unknown, use tf.cond with a symbolic comparison.
            return tf.cond(
                tf.equal(tf.rank(img), 4),
                lambda: tf.map_fn(self._augment_one, img, fn_output_signature=tf.uint8),
                lambda: self._augment_one(img)
            )

    def _apply_op(self, img, op_name, magnitude, fill):
        # Define a mapping from operation names to their corresponding functions
        def apply_shear_x(): return self.shear_x(img, magnitude)
        def apply_shear_y(): return self.shear_y(img, magnitude)
        def apply_translate_x(): return self.translate_x(img, magnitude)
        def apply_translate_y(): return self.translate_y(img, magnitude)
        def apply_rotate(): return self.rotate(img, magnitude)
        def apply_brightness(): return self.adjust_brightness(img, magnitude + 1.0)
        def apply_color(): return self.adjust_saturation(img, magnitude + 1.0)
        def apply_contrast(): return self.adjust_contrast(img, magnitude + 1.0)
        def apply_sharpness(): return self.adjust_sharpness(img, magnitude + 1.0)
        def apply_posterize(): return self.posterize(img, magnitude)
        def apply_solarize(): return self.solarize(img, magnitude)
        def apply_autocontrast(): return self.autocontrast(img)
        def apply_equalize(): return self.equalize(img)
        def apply_identity(): return img

        # Use tf.case to select the appropriate operation based on op_name
        img = tf.switch_case(
            branch_index=tf.cast(tf.argmax(tf.equal(op_name, tf.constant([
                "ShearX", "ShearY", "TranslateX", "TranslateY", "Rotate",
                "Brightness", "Color", "Contrast", "Sharpness", "Posterize",
                "Solarize", "AutoContrast", "Equalize", "Identity"
            ]))), tf.int32),
            branch_fns={
                0: apply_shear_x,
                1: apply_shear_y,
                2: apply_translate_x,
                3: apply_translate_y,
                4: apply_rotate,
                5: apply_brightness,
                6: apply_color,
                7: apply_contrast,
                8: apply_sharpness,
                9: apply_posterize,
                10: apply_solarize,
                11: apply_autocontrast,
                12: apply_equalize,
                13: apply_identity
            },
            default=apply_identity
        )
        return img

    def blend(self, image1, image2, factor):
        image1 = tf.cast(image1, tf.float32)
        image2 = tf.cast(image2, tf.float32)
        factor = tf.cast(factor, tf.float32)

        difference = image2 - image1
        scaled = factor * difference

        # Do addition in float.
        temp = image1 + scaled
        return tf.cast(tf.clip_by_value(temp, 0.0, 255.0), tf.uint8)

    # Affine Transformation Helper
    def _affine_transform(self, img, transform_matrix):
        a0, a1, a2 = transform_matrix[0], transform_matrix[1], transform_matrix[2]
        b0, b1, b2 = transform_matrix[3], transform_matrix[4], transform_matrix[5]

        # Create the 8-parameter transform tensor
        transforms = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])
        transforms = tf.reshape(transforms, [1, 8])  # Shape [1, 8]

        # Convert image to float32 and add batch dimension
        img_float = tf.cast(img, tf.float32)
        img_batched = tf.expand_dims(img_float, axis=0)

        # Apply affine transform using Keras ops
        transformed = tf.keras.ops.image.affine_transform(
            img_batched,
            transforms,
            interpolation='bilinear',
            fill_mode='constant',
            fill_value=self.fill if self.fill is not None else 0.0
        )
        # Remove batch dimension and convert back to uint8
        transformed = tf.squeeze(transformed, axis=0)
        return tf.cast(tf.clip_by_value(transformed, 0, 255), tf.uint8)

    def shear_x(self, img, magnitude):
        transform_matrix = tf.stack([
            1.0, magnitude, 0.0,  # a0, a1, a2
            0.0, 1.0, 0.0        # b0, b1, b2
        ])
        return self._affine_transform(img, transform_matrix)

    def shear_y(self, img, magnitude):
        transform_matrix = tf.stack([
            1.0, 0.0, 0.0,       # a0, a1, a2
            magnitude, 1.0, 0.0   # b0, b1, b2
        ])
        return self._affine_transform(img, transform_matrix)

    def translate_x(self, img, magnitude):
        transform_matrix = tf.stack([
            1.0, 0.0, magnitude,  # a0, a1, a2
            0.0, 1.0, 0.0         # b0, b1, b2
        ])
        return self._affine_transform(img, transform_matrix)

    def translate_y(self, img, magnitude):
        transform_matrix = tf.stack([
            1.0, 0.0, 0.0,       # a0, a1, a2
            0.0, 1.0, magnitude  # b0, b1, b2
        ])
        return self._affine_transform(img, transform_matrix)

    def rotate(self, img, angle):
        angle_rad = -math.pi * angle / 180.0
        cos_a = tf.cos(angle_rad)
        sin_a = tf.sin(angle_rad)

        # Center of rotation
        h, w = tf.shape(img)[0], tf.shape(img)[1]
        cx, cy = tf.cast(w, tf.float32) / 2.0, tf.cast(h, tf.float32) / 2.0

        transform_matrix = tf.stack([
            cos_a, -sin_a, (1 - cos_a) * cx + sin_a * cy,  # a0, a1, a2
            sin_a, cos_a, -sin_a * cx + (1 - cos_a) * cy   # b0, b1, b2
        ])
        return self._affine_transform(img, transform_matrix)

    # Adjust Brightness
    def adjust_brightness(self, img, brightness_factor):
        degenerate = tf.zeros_like(img)
        return self.blend(degenerate, img, brightness_factor)

    # Adjust Contrast
    def adjust_contrast(self, img, contrast_factor):
        # degenerate = tf.image.rgb_to_grayscale(img)
        # # Cast before calling tf.histogram.
        # degenerate = tf.cast(degenerate, tf.int32)

        # # Compute the grayscale histogram, then compute the mean pixel value,
        # # and create a constant image size of that value.  Use that as the
        # # blending degenerate target of the original image.
        # hist = tf.histogram_fixed_width(degenerate, [0, 255], nbins=256)
        # mean = tf.reduce_sum(tf.cast(hist, tf.float32)) / 256.0
        # degenerate = tf.ones_like(degenerate, dtype=tf.float32) * mean
        # degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
        # degenerate = tf.image.grayscale_to_rgb(tf.cast(degenerate, tf.uint8))
        degenerate = tf.math.reduce_mean(tf.image.rgb_to_grayscale(img), axis=[-3, -2, -1], keepdims=True)
        return self.blend(degenerate, img, contrast_factor)

    # Adjust Saturation
    def adjust_saturation(self, img, saturation_factor):
        degenerate = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(img))
        return self.blend(degenerate, img, saturation_factor)

    # Posterize
    def posterize(self, img, bits):
        bits = tf.cast(bits, img.dtype)
        shift = 8 - bits
        return tf.bitwise.left_shift(tf.bitwise.right_shift(img, shift), shift)

    # Solarize
    def solarize(self, img, threshold):
        threshold = tf.cast(threshold, img.dtype)
        return tf.where(img >= threshold, 255 - img, img)

    # Autocontrast
    def autocontrast(self, img):
        def scale_channel(image: tf.Tensor) -> tf.Tensor:
          """Scale the 2D image using the autocontrast rule."""
          # A possibly cheaper version can be done using cumsum/unique_with_counts
          # over the histogram values, rather than iterating over the entire image.
          # to compute mins and maxes.
          lo = tf.cast(tf.reduce_min(image), tf.float32)
          hi = tf.cast(tf.reduce_max(image), tf.float32)

          # Scale the image, making the lowest value 0 and the highest value 255.
          def scale_values(im):
            scale = 255.0 / (hi - lo)
            offset = -lo * scale
            im = tf.cast(im, tf.float32) * scale + offset
            im = tf.clip_by_value(im, 0.0, 255.0)
            return tf.cast(im, tf.uint8)

          result = tf.cond(hi > lo, lambda: scale_values(image), lambda: image)
          return result

        # Assumes RGB for now.  Scales each channel independently
        # and then stacks the result.
        s1 = scale_channel(img[..., 0])
        s2 = scale_channel(img[..., 1])
        s3 = scale_channel(img[..., 2])
        img = tf.stack([s1, s2, s3], -1)

        return img

    # Equalize
    def equalize(self, image):
        """Implements Equalize function from PIL using TF ops."""

        def scale_channel(im, c):
          """Scale the data in the channel to implement equalize."""
          im = tf.cast(im[..., c], tf.int32)
          # Compute the histogram of the image channel.
          histo = tf.histogram_fixed_width(im, [0, 255], nbins=256)

          # For the purposes of computing the step, filter out the nonzeros.
          nonzero = tf.where(tf.not_equal(histo, 0))
          nonzero_histo = tf.reshape(tf.gather(histo, nonzero), [-1])
          step = (tf.reduce_sum(nonzero_histo) - nonzero_histo[-1]) // 255

          def build_lut(histo, step):
            # Compute the cumulative sum, shifting by step // 2
            # and then normalization by step.
            lut = (tf.cumsum(histo) + (step // 2)) // step
            # Shift lut, prepending with 0.
            lut = tf.concat([[0], lut[:-1]], 0)
            # Clip the counts to be in range.  This is done
            # in the C code for image.point.
            return tf.clip_by_value(lut, 0, 255)

          # If step is zero, return the original image.  Otherwise, build
          # lut from the full histogram and step and then index from it.
          result = tf.cond(
              tf.equal(step, 0), lambda: im,
              lambda: tf.gather(build_lut(histo, step), im))

          return tf.cast(result, tf.uint8)

        # Assumes RGB for now.  Scales each channel independently
        # and then stacks the result.
        s1 = scale_channel(image, 0)
        s2 = scale_channel(image, 1)
        s3 = scale_channel(image, 2)
        image = tf.stack([s1, s2, s3], -1)
        return image

    def adjust_sharpness(self, image, factor):
        orig_image = image
        image = tf.cast(image, tf.float32)
        # Make image 4D for conv operation.
        image = tf.expand_dims(image, 0)
        # SMOOTH PIL Kernel.
        kernel = tf.constant([[1, 1, 1], [1, 5, 1], [1, 1, 1]],
                              dtype=tf.float32,
                              shape=[3, 3, 1, 1]) / 13.
        # Tile across channel dimension.
        kernel = tf.tile(kernel, [1, 1, 3, 1])
        strides = [1, 1, 1, 1]
        degenerate = tf.nn.depthwise_conv2d(
            image, kernel, strides, padding='VALID', dilations=[1, 1])
        degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
        degenerate = tf.squeeze(tf.cast(degenerate, tf.uint8), [0])

        # For the borders of the resulting image, fill in the values of the
        # original image.
        mask = tf.ones_like(degenerate)
        paddings = [[0, 0]] * (orig_image.shape.rank - 3)
        padded_mask = tf.pad(mask, paddings + [[1, 1], [1, 1], [0, 0]])
        padded_degenerate = tf.pad(degenerate, paddings + [[1, 1], [1, 1], [0, 0]])
        result = tf.where(tf.equal(padded_mask, 1), padded_degenerate, orig_image)

        # Blend the final result.
        return self.blend(result, orig_image, factor)
