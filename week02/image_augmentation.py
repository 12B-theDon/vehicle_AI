import cv2
import numpy as np
import os

# 저장할 디렉토리
output_dir = "homework_01"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

total_images_num = 1

# blur 필터 크기
blur_3 = (3, 3)
blur_5 = (5, 5)

# Noise
g_mean = 0
g_var = 0.5
g_sigma = g_var ** 0.5

for i in range(total_images_num):
    # Load the current image. 이미지 명: '{i}.jpg'.
    img_path = f'homework_01/{i}.jpg'
    cur_img = cv2.imread(img_path)
    if cur_img is None:
        print(f"Could not load image: {img_path}")
        continue

    cur_rows, cur_cols, cur_chs = cur_img.shape

    blur_idx = 0

    for j in range(2):
    # blur 분기
        if j == 0:
            blur = blur_3
            blur_idx = 3
        else:
            blur = blur_5
            blur_idx = 5
        curr_blur_image = cv2.blur(cur_img, blur)
        blur_filename = os.path.join(output_dir, f'image_{i}_blur_{blur_idx}.jpg')
        cv2.imwrite(blur_filename, curr_blur_image)
    # gaussian noise 분기
        for k in range(10):
            # Here we select a random std between 1 and 20 (inclusive)
            std = 1 + k * 2
            gauss_noise = np.random.normal(g_mean, std, (cur_rows, cur_cols, cur_chs))
            # Add noise to the blurred image and clip the values to valid range
            noised_img = curr_blur_image.astype(np.float32) + gauss_noise
            #noised_img = np.clip(noised_img, 0, 255).astype(np.uint8)
            noise_filename = os.path.join(output_dir, f'image_{i}_blur_{blur_idx}_noise_{k}.jpg')
            cv2.imwrite(noise_filename, noised_img)

    # translation 분기
            for l in range(10):
                # Define translation amounts (for example, shifting right and down by 'l' pixels)
                tx = l
                ty = l
                M_t = np.float32([[1, 0, tx], [0, 1, ty]])
                translated_img = cv2.warpAffine(noised_img, M_t, (cur_cols, cur_rows))
                trans_filename = os.path.join(output_dir, f'image_{i}_blur_{blur_idx}_noise_{k}_trans_{l}.jpg')
                cv2.imwrite(trans_filename, translated_img)

    # rotation 분기
                # Apply rotation transformations (11 variations from -5 to +5 degrees)
                for m in range(11):
                    angle = -5 + m  # will vary from -5 to +5
                    M_r = cv2.getRotationMatrix2D((cur_cols / 2, cur_rows / 2), angle, 1)
                    rotated_img = cv2.warpAffine(translated_img, M_r, (cur_cols, cur_rows))
                    rot_filename = os.path.join(
                        output_dir, f'image_{i}_blur_{blur_idx}_noise_{k}_trans_{l}_rot_{angle}.jpg'
                    )
                    cv2.imwrite(rot_filename, rotated_img)

