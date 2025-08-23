import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PILImage

import os
os.makedirs('save_here6', exist_ok=True)

# Load VGG19 model
model = tf.keras.applications.VGG19(include_top=False,
                                    input_shape=(400, 400, 3),
                                    weights='imagenet')
model.trainable = False

STYLE_LAYERS = [
    'block1_conv1',
    'block2_conv1', 
    'block3_conv1',
    'block4_conv1',
    'block5_conv1'
]
CONTENT_LAYER = 'block5_conv4'


def get_model_outputs(vgg, style_layers, content_layer):
    """Extract outputs from specified layers"""
    style_outputs = [vgg.get_layer(layer).output for layer in style_layers]
    content_output = vgg.get_layer(content_layer).output
    outputs = style_outputs + [content_output]
    return tf.keras.Model([vgg.input], outputs)

# Create the model for extracting features
feature_extractor = get_model_outputs(model, STYLE_LAYERS, CONTENT_LAYER)

def gram_matrix(input_tensor):
    """Calculate Gram matrix for style representation"""
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations

def style_loss(style_targets, style_outputs):
    """Calculate style loss"""
    loss = 0
    for target, output in zip(style_targets, style_outputs):
        target_gram = gram_matrix(target)
        output_gram = gram_matrix(output)
        loss += tf.reduce_mean(tf.square(target_gram - output_gram))
    return loss / len(style_targets)

def content_loss(content_target, content_output):
    """Calculate content loss"""
    return tf.reduce_mean(tf.square(content_target - content_output))

def total_variation_loss(image):
    """Reduce high frequency noise"""
    x_deltas = image[:, :, 1:, :] - image[:, :, :-1, :]
    y_deltas = image[:, 1:, :, :] - image[:, :-1, :, :]
    return tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))

def preprocess_image(image_path, target_size=(400, 400)):
    """Load and preprocess image"""
    img = tf.keras.utils.load_img(image_path, target_size=target_size)
    img = tf.keras.utils.img_to_array(img)
    img = tf.expand_dims(img, axis=0)
    return tf.keras.applications.vgg19.preprocess_input(img)

def deprocess_image(processed_img):
    """Convert VGG preprocessed image back to displayable format"""
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    
    # BGR to RGB
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# Load and preprocess images
content_image = preprocess_image("images/louvre.jpg")
style_image = preprocess_image("images/monet.jpg")

# Get target features
style_targets = feature_extractor(style_image)[:-1]  # All but last (content)
content_target = feature_extractor(content_image)[-1]  # Last layer (content)

# Initialize generated image
generated_image = tf.Variable(content_image, trainable=True)

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.02)

# Loss weights - INCREASED style, REDUCED TV for more texture
style_weight = 7e-2  # 5x stronger style transfer
content_weight = 1e4
tv_weight = 10       # Much lower TV to preserve detail

@tf.function
def train_step():
    with tf.GradientTape() as tape:
        outputs = feature_extractor(generated_image)
        style_outputs = outputs[:-1]
        content_output = outputs[-1]
        
        s_loss = style_loss(style_targets, style_outputs) * style_weight
        c_loss = content_loss(content_target, content_output) * content_weight
        tv_loss = total_variation_loss(generated_image) * tv_weight
        
        total_loss = s_loss + c_loss + tv_loss
        
    gradients = tape.gradient(total_loss, generated_image)
    optimizer.apply_gradients([(gradients, generated_image)])
    
    # Clip to valid range
    generated_image.assign(tf.clip_by_value(generated_image, 
                                           clip_value_min=-103.939, 
                                           clip_value_max=255-123.68))
    
    return total_loss, s_loss, c_loss, tv_loss

# Training loop
epochs = 20000
display_interval = 1000

print("Starting Neural Style Transfer...")
print(f"Content weight: {content_weight}, Style weight: {style_weight}, TV weight: {tv_weight}")

for epoch in range(epochs + 1):
    total_loss_val, style_loss_val, content_loss_val, tv_loss_val = train_step()
    
    if epoch % display_interval == 0:
        print(f"\nEpoch {epoch}:")
        print(f"  Total loss: {total_loss_val:.2e}")
        print(f"  Content loss: {content_loss_val:.2e}")
        print(f"  Style loss: {style_loss_val:.2e}")
        print(f"  TV loss: {tv_loss_val:.2e}")
        
        # Display current result
        current_image = deprocess_image(generated_image.numpy())
        
        plt.figure(figsize=(12, 4))
        
        # Original content
        plt.subplot(1, 3, 1)
        plt.imshow(deprocess_image(content_image.numpy()))
        plt.title('Content Image')
        plt.axis('off')
        
        # Style image
        plt.subplot(1, 3, 2)
        plt.imshow(deprocess_image(style_image.numpy()))
        plt.title('Style Image')
        plt.axis('off')
        
        # Current result
        plt.subplot(1, 3, 3)
        plt.imshow(current_image)
        plt.title(f'Generated (Epoch {epoch})')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Save image
        pil_img = PILImage.fromarray(current_image)
        pil_img.save(f'save_here6/nst_epoch_{epoch:04d}.jpg')

print("\n🎉 Training Complete!")