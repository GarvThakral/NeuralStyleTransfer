import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def preprocess_pil_image(pil_img, target_size=(400, 400)):
    img = pil_img.resize(target_size)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.expand_dims(img, axis=0)
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img

def gram_matrices_vectorized(a_a):
    b, i, j, k = a_a.shape
    X = tf.reshape(a_a, (b, i*j, k))
    gram_matrix = tf.matmul(X, X, transpose_a=True)
    return gram_matrix

def style_cost(G_s, G_g, a_s):
    batch_size, n_h, n_w, n_c = a_s.shape
    sub = tf.reduce_sum(tf.square(tf.subtract(G_s, G_g)))
    norm = 4.0 * tf.cast(n_h * n_w, tf.float32) * tf.cast(n_c * n_c, tf.float32)
    return tf.divide(sub, norm)

def content_cost(a_c, a_g_c):
    return 0.5 * tf.reduce_mean(tf.square(a_c - a_g_c))

def total_variation_loss(img):
    x_deltas = img[:, :, 1:, :] - img[:, :, :-1, :]
    y_deltas = img[:, 1:, :, :] - img[:, :-1, :, :]
    return tf.reduce_mean(tf.abs(x_deltas)) + tf.reduce_mean(tf.abs(y_deltas))

def vgg_to_display(vgg_tensor):
    img = vgg_tensor.numpy()
    img = img + [103.939, 116.779, 123.68]
    img = img[..., ::-1]  # BGR to RGB
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def your_generation_function(image1, image2):
    model = tf.keras.applications.VGG19(include_top=False,
                                        input_shape=(400, 400, 3),
                                        weights='imagenet')
    model.trainable = False

    STYLE_LAYERS = [
        ('block1_conv1', 0.2),
        ('block2_conv1', 0.2),
        ('block3_conv1', 0.2),
        ('block4_conv1', 0.2),
        ('block5_conv1', 0.2)
    ]
    CONTENT_LAYER = [('block5_conv4', 1.0)]

    def get_layer_outputs(vgg, layer_names):
        outputs = [vgg.get_layer(layer[0]).output for layer in layer_names]
        return tf.keras.Model([vgg.input], outputs)

    vgg_model_outputs = get_layer_outputs(model, STYLE_LAYERS + CONTENT_LAYER)

    styleImg = preprocess_pil_image(image1)
    contentImg = preprocess_pil_image(image2)
    generatedImg = tf.Variable(contentImg, trainable=True)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.08)

    epochs = 2000
    display_interval = 5

    # For Streamlit display
    st_placeholder = st.empty()

    for i in range(epochs + 1):
        print(i)
        with tf.GradientTape() as tape:
            generated_vgg = generatedImg

            all_outputs = vgg_model_outputs(styleImg)
            a_s, a_s_1, a_s_2, a_s_3, a_s_4, a_c = all_outputs

            all_outputs_gen = vgg_model_outputs(generated_vgg)
            a_g_s, a_g_s_1, a_g_s_2, a_g_s_3, a_g_s_4, a_g_c = all_outputs_gen

            style_layers_target = [a_s, a_s_1, a_s_2, a_s_3, a_s_4]
            style_layers_gen = [a_g_s, a_g_s_1, a_g_s_2, a_g_s_3, a_g_s_4]

            style_loss_total = 0
            for target, gen in zip(style_layers_target, style_layers_gen):
                G_s = gram_matrices_vectorized(target)
                G_g = gram_matrices_vectorized(gen)
                style_loss_total += style_cost(G_s, G_g, target)
            style_loss_total /= len(style_layers_target)

            content_loss_val = content_cost(a_c, a_g_c)

            alpha = 1e-1
            beta = 8e3
            tv_weight = 3

            main_loss = alpha * content_loss_val + beta * style_loss_total
            tv_loss = total_variation_loss(generatedImg) * tv_weight
            loss = main_loss + tv_loss

        grads = tape.gradient(loss, generatedImg)
        optimizer.apply_gradients([(grads, generatedImg)])

        generatedImg.assign(tf.clip_by_value(generatedImg,
                                             clip_value_min=-103.939,
                                             clip_value_max=255 - 123.68))

        if i % display_interval == 0:
            display_img = vgg_to_display(generatedImg[0])

            st_placeholder.image(display_img, caption=f'Iteration {i}', use_column_width=True)

            print(f"\nIteration {i}:")
            print(f"Total loss: {loss.numpy():.2e}")
            print(f"  Content loss: {content_loss_val.numpy():.2e}")
            print(f"  Style loss: {style_loss_total.numpy():.2e}")
            print(f"  TV loss: {tv_loss.numpy():.2e}")
            print(f"  Generated img range: [{tf.reduce_min(generatedImg).numpy():.3f}, {tf.reduce_max(generatedImg).numpy():.3f}]")
            print(f"  Display img stats: Min={display_img.min()}, Max={display_img.max()}")
            print(f"  Mean RGB: {display_img.mean(axis=(0, 1))}")

    print("\n🎉 Training Complete!")

    final_img = vgg_to_display(generatedImg[0])
    result_img = Image.fromarray(final_img)
    return result_img


def main():
    st.title("Image Generator")
    st.write("Upload two images and generate a third one!")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Image 1")
        uploaded_file1 = st.file_uploader("Choose first image", type=['png', 'jpg', 'jpeg'], key="img1")
        if uploaded_file1:
            image1 = Image.open(uploaded_file1)
            st.image(image1, caption="Image 1", use_column_width=True)

    with col2:
        st.subheader("Image 2")
        uploaded_file2 = st.file_uploader("Choose second image", type=['png', 'jpg', 'jpeg'], key="img2")
        if uploaded_file2:
            image2 = Image.open(uploaded_file2)
            st.image(image2, caption="Image 2", use_column_width=True)

    with col3:
        st.subheader("Generated Image")
        if 'image1' in locals() and 'image2' in locals():
            if st.button("Generate", type="primary"):
                with st.spinner("Generating... This may take a while!"):
                    generated_image = your_generation_function(image1, image2)
                    st.image(generated_image, caption="Generated Image", use_column_width=True)
        else:
            st.info("Upload both images to enable generation")


if __name__ == "__main__":
    main()
