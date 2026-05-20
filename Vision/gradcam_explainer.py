import cv2
import numpy as np
import tensorflow as tf
from PIL import Image


def generate_gradcam(model: tf.keras.Model, image_tensor: np.ndarray, target_layer: str) -> np.ndarray:
    """Compute a Grad-CAM saliency heatmap for the top predicted class.

    Builds an auxiliary model that exposes both the target convolutional layer's
    output and the final predictions. Differentiates the top-class score with
    respect to the convolutional activations, pools the gradients spatially, and
    weights each activation map accordingly (Section 2.5).

    Args:
        model:        Trained Keras model.
        image_tensor: Preprocessed input, shape (1, H, W, 3), float32.
        target_layer: Name of the convolutional layer to visualise.

    Returns:
        2-D float32 heatmap in [0, 1], spatial resolution of target_layer.
    """
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(target_layer).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(tf.cast(image_tensor, tf.float32))
        tape.watch(conv_outputs)
        top_class = int(tf.argmax(predictions[0]))
        class_score = predictions[:, top_class]

    grads = tape.gradient(class_score, conv_outputs)          # (1, h, w, C)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))     # (C,)

    # Weight each activation map by its importance and sum across channels.
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]  # (h, w, 1)
    heatmap = tf.squeeze(heatmap)                               # (h, w)
    heatmap = tf.nn.relu(heatmap)

    max_val = tf.reduce_max(heatmap)
    heatmap = heatmap / (max_val + 1e-8)

    return heatmap.numpy().astype(np.float32)


def overlay_heatmap(original: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> Image.Image:
    """Blend a Grad-CAM heatmap onto the original image.

    Resizes the heatmap to match the original image, applies the JET colormap,
    and composites it over the original using addWeighted (Section 2.5).

    Args:
        original: BGR uint8 image (H, W, 3).
        heatmap:  2-D float32 array in [0, 1] from generate_gradcam.
        alpha:    Heatmap blend weight (default 0.4).

    Returns:
        PIL Image (RGB) with the coloured heatmap overlaid.
    """
    h, w = original.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap_uint8 = np.uint8(255.0 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(original, 1.0 - alpha, heatmap_colored, alpha, 0)
    return Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
