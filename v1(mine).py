dim = 400  

# After loading images, convert to float32
img1 = tf.keras.utils.load_img("images/monet.jpg", target_size=(dim,dim))
img1 = tf.keras.utils.img_to_array(img1)
img1 = tf.expand_dims(img1, axis=0)
img1 = tf.cast(img1, tf.float32)
styleImg = tf.keras.applications.vgg19.preprocess_input(img1)

img2 = tf.keras.utils.load_img("images/louvre.jpg", target_size=(dim,dim))
img2 = tf.keras.utils.img_to_array(img2)
img2 = tf.expand_dims(img2, axis=0)
img2 = tf.cast(img2, tf.float32)
contentImg = tf.keras.applications.vgg19.preprocess_input(img2)

generatedImg = tf.Variable(contentImg, trainable=True)

def generated_to_display(generated_tensor):
    img = generated_tensor.numpy() * 255
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def vgg_to_display(vgg_tensor):
    img = vgg_tensor.numpy()
    img = img + [103.939, 116.779, 123.68]
    img = img[..., ::-1]
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

optimizer = tf.keras.optimizers.Adam(learning_rate=0.02)  # Fixed LR

for i in range(20001):
    with tf.GradientTape() as tape:
        generated_vgg = generatedImg
        
        all_outputs = vgg_model_outputs(styleImg)
        a_s, a_s_1, a_s_2, a_s_3, a_s_4, a_c = all_outputs

        all_outputs_gen = vgg_model_outputs(generated_vgg) 
        a_g_s, a_g_s_1, a_g_s_2, a_g_s_3, a_g_s_4, a_g_c = all_outputs_gen
        
        # Calculate style loss over all 5 style layers like second code
        style_layers_target = [a_s, a_s_1, a_s_2, a_s_3, a_s_4]
        style_layers_gen = [a_g_s, a_g_s_1, a_g_s_2, a_g_s_3, a_g_s_4]
        
        style_loss_total = 0
        for target, gen in zip(style_layers_target, style_layers_gen):
            G_s = gram_matrices_vectorized(target)
            G_g = gram_matrices_vectorized(gen)
            style_loss_total += style_cost(G_s, G_g, target)
        style_loss_total /= len(style_layers_target)
        
        content_loss_val = content_cost(a_c, a_g_c)
        
        # Adjust loss weights to approximate second code
        alpha = 1e4       # content weight
        beta = 7e-2       # style weight
        tv_weight = 10
        
        main_loss = alpha * content_loss_val + beta * style_loss_total
        
        tv_loss = total_variation_loss(generatedImg) * tv_weight
        loss = main_loss + tv_loss
        
        grads = tape.gradient(loss, generatedImg)
        # Removed gradient normalization and clipping to match second code
        
        optimizer.apply_gradients([(grads, generatedImg)])
        
        generatedImg.assign(tf.clip_by_value(generatedImg, 
                                             clip_value_min=-103.939, 
                                             clip_value_max=255-123.68))
        
        # Adjusted display interval to 1000
        if i % 1000 == 0:
            print(f"\nIteration {i}:")
            print(f"Total loss: {loss.numpy():.2e}")
            print(f"  Content loss: {content_loss_val.numpy():.2e}")
            print(f"  Style loss: {style_loss_total.numpy():.2e}")
            print(f"  TV loss: {tv_loss.numpy():.2e}")
            print(f"  Generated img range: [{tf.reduce_min(generatedImg).numpy():.3f}, {tf.reduce_max(generatedImg).numpy():.3f}]")
            
            display_img = vgg_to_display(generatedImg[0])
            print(f"  Display img stats: Min={display_img.min()}, Max={display_img.max()}")
            print(f"  Mean RGB: {display_img.mean(axis=(0,1))}")
            
            plt.figure(figsize=(10, 10))
            plt.imshow(display_img)
            plt.title(f'Neural Style Transfer - Iteration {i}')
            plt.axis('off')
            plt.show()
            
            tf.keras.utils.save_img(
                f'./save_here6/blue_image_{i:05d}.jpeg', 
                display_img, 
                file_format='jpeg'
            )
            
print("\n🎉 Training Complete!")
final_display = vgg_to_display(generatedImg)
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(vgg_to_display(contentImg))
plt.title('Original Content')
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(vgg_to_display(styleImg))
plt.title('Style Image')
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(final_display)
plt.title('Generated Result')
plt.axis('off')
plt.suptitle('Neural Style Transfer Results (Modified)', fontsize=16)
plt.tight_layout()
plt.show()
