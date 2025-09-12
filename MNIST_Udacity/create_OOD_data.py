'''
Constructing OOD testing sets for MNIST and Udacity datasets studied in this paper.
'''
import tensorflow as tf
import numpy as np
import os
import keras
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import random
from glob import glob
from PIL import Image
from tqdm import tqdm
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing import image
basedir = os.path.abspath(os.path.dirname(__file__))
'''
For attack parameters, we set
eps=8/255 for udacity driving;
eps=0.3 for MNIST dataset
'''

iterations = 10  # Number of iterations for BIM and PGD

# FGSM attack
def fgsm_attack(model, images, labels, epsilon):
    images = tf.convert_to_tensor(images)
    labels = tf.convert_to_tensor(labels)
    with tf.GradientTape() as tape:
        tape.watch(images)
        predictions = model(images)
        loss = tf.keras.losses.categorical_crossentropy(labels, predictions)
    gradient = tape.gradient(loss, images)
    import pdb; pdb.set_trace()
    signed_grad = tf.sign(gradient)
    adversarial_images = images + epsilon * signed_grad
    adversarial_images = tf.clip_by_value(adversarial_images, 0, 1)
    return adversarial_images.numpy()

def preprocess_image(img_path, target_size=(100, 100)):
    img = image.load_img(img_path, target_size=target_size)
    input_img_data = image.img_to_array(img)
    input_img_data = np.expand_dims(input_img_data, axis=0)
    input_img_data = preprocess_input(input_img_data)
    return input_img_data

def fgsm_driving_attack(model, paths, labels, epsilon):
    adv_imgs = []
    for path, label in zip(paths, labels):
        image = preprocess_image(path, (100, 100))
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        label = tf.convert_to_tensor(label, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(image)
            prediction = model(image)
            loss = tf.keras.losses.mean_squared_error(label, prediction)
            loss = tf.reduce_mean(loss)  # Ensure it's a scalar
        gradient = tape.gradient(loss, image)
        signed_grad = tf.sign(gradient)
        adversarial_image = image + epsilon * 255 * signed_grad
        adversarial_image = deprocess_image(adversarial_image.numpy())
        adv_imgs.append(adversarial_image)
    return np.array(adv_imgs) # return un-processed imgs, original scale

# BIM attack
def bim_driving_attack(model, paths, labels, epsilon, alpha, iterations):
    store_adv_imgs = []
    for path, label in zip(paths, labels):
        image = preprocess_image(path, (100, 100))
        image_ori = image.copy()
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        label = tf.convert_to_tensor(label, dtype=tf.float32)

        for _ in range(iterations):
            with tf.GradientTape() as tape:
                tape.watch(image)
                prediction = model(image)
                loss = tf.keras.losses.mean_squared_error(label, prediction)
                loss = tf.reduce_mean(loss)  # Scalar loss required
            gradient = tape.gradient(loss, image)
            signed_grad = tf.sign(gradient)
            image = image + alpha * 255 * signed_grad
            image = tf.clip_by_value(image, image_ori - epsilon * 255, image_ori + epsilon * 255)
        adversarial_image = deprocess_image(image.numpy())
        store_adv_imgs.append(adversarial_image)
    return np.array(store_adv_imgs)

def pgd_driving_attack(model, paths, labels, epsilon, alpha, iterations):
    adv_imgs = []

    for path, label in zip(paths, labels):
        image = preprocess_image(path, (100, 100))  # shape: (1, 100, 100, 3)
        image_ori = image.copy()

        # Add initial random noise within epsilon ball
        adv_image = image + np.random.uniform(-epsilon * 255, epsilon * 255, image.shape).astype(np.float32)
        label_tensor = tf.convert_to_tensor(label, dtype=tf.float32)

        for _ in range(iterations):
            adv_tensor = tf.convert_to_tensor(adv_image, dtype=tf.float32)
            with tf.GradientTape() as tape:
                tape.watch(adv_tensor)
                prediction = model(adv_tensor)
                loss = tf.keras.losses.mean_squared_error(label_tensor, prediction)
                loss = tf.reduce_mean(loss)

            gradient = tape.gradient(loss, adv_tensor)
            signed_grad = tf.sign(gradient)
            adv_image = adv_image + alpha * 255 * signed_grad.numpy()

            # Project back to epsilon-ball
            adv_image = np.clip(adv_image, image_ori - epsilon * 255, image_ori + epsilon * 255)

        # Convert to uint8 RGB image for visualization
        adversarial_img = deprocess_image(adv_image)
        adv_imgs.append(adversarial_img)

    return np.array(adv_imgs)

def bim_attack(model, images, labels, epsilon, alpha, iterations):
    adv_images = images.copy()
    for _ in range(iterations):
        adv_images = tf.convert_to_tensor(adv_images)
        labels = tf.convert_to_tensor(labels)
        with tf.GradientTape() as tape:
            tape.watch(adv_images)
            predictions = model(adv_images)
            loss = tf.keras.losses.categorical_crossentropy(labels, predictions)
        gradient = tape.gradient(loss, adv_images)
        signed_grad = tf.sign(gradient)
        adv_images = adv_images + alpha * signed_grad
        adv_images = tf.clip_by_value(adv_images, images - epsilon, images + epsilon)
        adv_images = tf.clip_by_value(adv_images, 0, 1)
        adv_images = adv_images.numpy()
    return adv_images
# PGD attack
def pgd_attack(model, images, labels, epsilon, alpha, iterations):
    adv_images = images + np.random.uniform(-epsilon, epsilon, images.shape)
    adv_images = np.clip(adv_images, 0, 1)
    for _ in range(iterations):
        adv_images_tensor = tf.convert_to_tensor(adv_images)
        labels_tensor = tf.convert_to_tensor(labels)
        with tf.GradientTape() as tape:
            tape.watch(adv_images_tensor)
            predictions = model(adv_images_tensor)
            loss = tf.keras.losses.categorical_crossentropy(labels_tensor, predictions)
        gradient = tape.gradient(loss, adv_images_tensor)
        signed_grad = tf.sign(gradient)
        adv_images = adv_images + alpha * signed_grad.numpy()
        adv_images = np.clip(adv_images, images - epsilon, images + epsilon)
        adv_images = np.clip(adv_images, 0, 1)
    return adv_images

def create_corrupted_mnist():
    basedir = os.path.abspath(os.path.dirname(__file__))
    model = keras.models.load_model(os.path.join(basedir, 'model/LeNet-5.h5'))  # we attack lenet5

    (_, _), (X_test, Y_test) = mnist.load_data()
    X_test = X_test.astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    X_test /= 255
    y_test = keras.utils.to_categorical(Y_test, 10)

    # Select 5,000 random indices for corrupted examples
    np.random.seed(123)  # For reproducibility
    adv_indices = np.random.choice(len(X_test), size=5000, replace=False)

    corrupted_X = np.load(os.path.join(basedir,'data/corrupted_image/mnist_c_images.npy'))
    corrupted_Y = np.load(os.path.join(basedir, 'data/corrupted_image/mnist_c_labels.npy'))
    corrupted_X = corrupted_X.astype(np.float32) / 255.0
    corrupted_y = keras.utils.to_categorical(corrupted_Y, 10)

    all_indices = set(range(len(X_test)))
    clean_indices = np.array(list(all_indices - set(adv_indices)))
    # Extract clean images and labels
    X_clean = X_test[clean_indices]
    y_clean = y_test[clean_indices]

    # Combine adversarial and clean examples
    X_combined = np.concatenate([corrupted_X[adv_indices], X_clean], axis=0)
    y_combined = np.concatenate([corrupted_y[adv_indices], y_clean], axis=0)
    # Shuffle the combined dataset
    combined_indices = np.arange(len(X_combined))
    np.random.shuffle(combined_indices)
    X_combined = X_combined[combined_indices]
    y_combined = y_combined[combined_indices]

    np.save(os.path.join(basedir, 'data/corrupted_image/corrupted_clean_mnist_image.npy'), X_combined)
    np.save(os.path.join(basedir, 'data/corrupted_image/corrupted_clean_mnist_label.npy'), y_combined)


def create_adv_mnist():
    epsilon = 0.3
    alpha = epsilon / 10  # Step size for iterative attacks
    model = keras.models.load_model(os.path.join(basedir, 'mnist_cifar_imagenet_svhn/model/LeNet-5.h5')) #we attack lenet5

    (_, _), (X_test, Y_test) = mnist.load_data()
    X_test = X_test.astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    X_test /= 255
    y_test = keras.utils.to_categorical(Y_test, 10)

    # Select 5,000 random indices for adversarial examples
    np.random.seed(123)  # For reproducibility
    adv_indices = np.random.choice(len(X_test), size=5000, replace=False)

    # Split indices for each attack
    split_size = 5000 // 3
    fgsm_indices = adv_indices[:split_size]
    bim_indices = adv_indices[split_size:2 * split_size]
    pgd_indices = adv_indices[2 * split_size:]

    # Generate adversarial examples
    X_fgsm_adv = fgsm_attack(model, X_test[fgsm_indices], y_test[fgsm_indices], epsilon)
    X_bim_adv = bim_attack(model, X_test[bim_indices], y_test[bim_indices], epsilon, alpha, iterations)
    X_pgd_adv = pgd_attack(model, X_test[pgd_indices], y_test[pgd_indices], epsilon, alpha, iterations)

    # Combine adversarial examples and labels
    X_adv = np.concatenate([X_fgsm_adv, X_bim_adv, X_pgd_adv], axis=0)
    y_adv = np.concatenate([y_test[fgsm_indices], y_test[bim_indices], y_test[pgd_indices]], axis=0)

    # Determine clean indices (excluding adversarial ones)
    all_indices = set(range(len(X_test)))
    clean_indices = np.array(list(all_indices - set(adv_indices)))

    # Extract clean images and labels
    X_clean = X_test[clean_indices]
    y_clean = y_test[clean_indices]

    # Combine adversarial and clean examples
    X_combined = np.concatenate([X_adv, X_clean], axis=0)
    y_combined = np.concatenate([y_adv, y_clean], axis=0)

    # Shuffle the combined dataset
    combined_indices = np.arange(len(X_combined))
    np.random.shuffle(combined_indices)
    X_combined = X_combined[combined_indices]
    y_combined = y_combined[combined_indices]

    np.save(os.path.join(basedir, 'data/adv_image/fgsm_bim_pgd_clean_mnist_image.npy'), X_combined)
    np.save(os.path.join(basedir, 'data/adv_image/fgsm_bim_pgd_clean_mnist_label.npy'), y_combined)


def create_mnist_label():
    (_, _), (X_test, Y_test) = mnist.load_data()
    X_test = X_test.astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    X_test /= 255
    y_test = to_categorical(Y_test, 10)
    # Desired long-tailed label distribution
    distribution = {
        0: 0.34,
        1: 0.15,
        2: 0.15,
        3: 0.15,
        4: 0.05,
        5: 0.05,
        6: 0.05,
        7: 0.02,
        8: 0.02,
        9: 0.02
    }
    X_new, y_new = [], []
    n_total = X_test.shape[0]
    np.random.seed(42)

    for digit, frac in distribution.items():
        count = int(frac * n_total)
        indices = np.where(Y_test == digit)[0]
        chosen = np.random.choice(indices, size=count, replace=(count > len(indices)))
        X_new.append(X_test[chosen])
        y_new.append(y_test[chosen])

    # Combine and shuffle
    X_mnist_label = np.concatenate(X_new)
    Y_mnist_label = np.concatenate(y_new)
    perm = np.random.permutation(len(X_mnist_label))
    X_mnist_label = X_mnist_label[perm]
    Y_mnist_label = Y_mnist_label[perm]

    # Save to .npy files
    np.save("mnist_label_imgs.npy", X_mnist_label)
    np.save("mnist_label_labels.npy", Y_mnist_label)

    # Print sample counts for verification
    digit_counts = {i: int(np.sum(np.argmax(Y_mnist_label, axis=1) == i)) for i in range(10)}
    # {0: 3400, 1: 1500, 2: 1500, 3: 1500, 4: 500, 5: 500, 6: 500, 7: 200, 8: 200, 9: 200}
    print("Sample count per digit (MNIST-label):", digit_counts)


def create_mnist_emnist():
    (_, _), (X_mnist, Y_mnist) = mnist.load_data()
    X_mnist = X_mnist.astype('float32').reshape(-1, 28, 28, 1) / 255.0
    Y_mnist = Y_mnist.astype('int32')

    import tensorflow_datasets as tfds
    ds_emnist = tfds.load('emnist/mnist', split='test', as_supervised=True)
    X_emnist, Y_emnist = [], []
    for x, y in tfds.as_numpy(ds_emnist):
        X_emnist.append(x)
        Y_emnist.append(y)
    X_emnist = np.array(X_emnist).astype('float32') / 255.0
    Y_emnist = np.array(Y_emnist).astype('int32')

    # EMNIST is rotated, so rotate it to match MNIST orientation
    X_emnist = np.transpose(X_emnist, (0, 2, 1, 3))  # rotate 90 degrees

    np.random.seed(42)
    n_samples = 5000

    mnist_idx = np.random.choice(len(X_mnist), n_samples, replace=False)
    emnist_idx = np.random.choice(len(X_emnist), n_samples, replace=False)

    X_mnist_sub = X_mnist[mnist_idx]
    Y_mnist_sub = Y_mnist[mnist_idx]
    X_emnist_sub = X_emnist[emnist_idx]
    Y_emnist_sub = Y_emnist[emnist_idx]
    X_combined = np.concatenate([X_mnist_sub, X_emnist_sub], axis=0)
    Y_combined = np.concatenate([Y_mnist_sub, Y_emnist_sub], axis=0)

    perm = np.random.permutation(len(X_combined))
    X_combined = X_combined[perm]
    Y_combined = Y_combined[perm]
    np.save('mnist_emnist_imgs.npy', X_combined)
    np.save('mnist_emnist_labels.npy', Y_combined)

def create_udacity_c():
    import imagecorruptions
    # ==== Config ====
    input_dir = os.path.join(basedir, 'driving/testing/center')
    output_base = os.path.join(basedir, 'driving/data/udacity_output_corrupted')
    image_size = (640, 480)  # width, height
    corruptions = imagecorruptions.get_corruption_names()
    severity_levels = [1, 2, 3, 4, 5]

    # Read and shuffle all image paths
    all_images = sorted(glob(os.path.join(input_dir, "*.jpg")))
    random.shuffle(all_images)

    num_corruptions = len(corruptions)
    images_per_corruption = len(all_images) // num_corruptions
    leftover = len(all_images) % num_corruptions

    start_idx = 0
    for i, corruption in enumerate(corruptions): #15 corruptions
        # Determine number of images for this corruption
        num_images = images_per_corruption + (1 if i < leftover else 0)
        corruption_images = all_images[start_idx:start_idx + num_images]
        start_idx += num_images

        # Output folder for this corruption
        output_dir = os.path.join(output_base, corruption)
        os.makedirs(output_dir, exist_ok=True)

        for j, img_path in enumerate(corruption_images):
            severity = random.choice(severity_levels)
            with Image.open(img_path) as img:

                #img = img.resize(image_size).convert("RGB")
                img_bytes = imagecorruptions.corrupt(np.array(img), corruption_name=corruption, severity=severity)
                corrupted_img = Image.fromarray(img_bytes)

                filename = f"{img_path.split('/')[-1].split('.')[0]}.jpg"
                corrupted_img.save(os.path.join(output_dir, filename))

def create_udacity_c_combined():
    '''
    50% corrupted and 50% clean images, total 5614 images
    '''
    input_dir = os.path.join(basedir, 'driving/testing/center')
    all_images = sorted(glob(os.path.join(input_dir, "*.jpg")))
    # Select 5,000 random indices for adversarial examples
    np.random.seed(123)  # For reproducibility

    # Parameters
    base_dir = os.path.join(basedir, 'driving/data/udacity_output_corrupted')
    num_samples = 5614 // 2
    selected_filenames = []

    # Collect all .jpg paths from all corruption folders
    all_image_paths = []
    corruption_folders = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if
                          os.path.isdir(os.path.join(base_dir, d))]

    for folder in corruption_folders:
        image_paths = glob(os.path.join(folder, "*.jpg"))
        all_image_paths.extend(image_paths)

    # Shuffle and select images
    random.shuffle(all_image_paths)
    selected_images = all_image_paths[:num_samples]

    # Extract just the filenames
    selected_filenames = [os.path.basename(path) for path in selected_images]

    # Create a set for faster lookup
    excluded_basenames = set(selected_filenames)
    # Filter all_images to exclude those whose basename is in selected_filenames
    remaining_images = [path for path in all_images if os.path.basename(path) not in excluded_basenames]

    # Combine both lists
    all_combined_images = remaining_images + selected_images
    # Save to a text file
    output_file = "Udacity_C_clean.txt"
    with open(output_file, "w") as f:
        for path in all_combined_images:
            f.write(f"{path}\n")

    print(f"Saved {len(all_combined_images)} image paths to {output_file}")

def create_udacity_dave():
    TOTAL_SAMPLES = 5614
    SAMPLES_PER_DATASET = TOTAL_SAMPLES // 2  # 2807 from each

    dave_xs = []
    dave_ys = []
    # 45406 images
    with open("./dave_test/data.txt") as f:
        for line in tqdm(f):
            dave_xs.append("./dave_test/driving_dataset/" + line.split()[0])
            # ys is outputted in radians. *-1 to be consistent w/ udacity (pos=left, neg=right)
            steering_ratio = 15  # or use 16, 18 depending on car type
            steering_angle_wheel_deg = float(line.split()[1]) / steering_ratio
            steering_angle = -1 * float(steering_angle_wheel_deg) * 3.14159265 / 180
            dave_ys.append(steering_angle)

    dave_xs, dave_ys = dave_xs[:SAMPLES_PER_DATASET], dave_ys[:SAMPLES_PER_DATASET]

    # load udacity testing data
    input_dir = os.path.join(basedir, 'driving/testing/')
    csv_input = input_dir + 'CH2_final_evaluation.csv'  # Input CSV path
    img_folder = input_dir + 'center/'  # Folder prefix for images if needed

    # === Load original data ===
    udacity_xs, udacity_ys = [], []
    with open(csv_input, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            udacity_xs.append(img_folder + line.split(',')[0] + '.jpg')
            udacity_ys.append(float(line.split(',')[1]))

    udacity_xs, udacity_ys = udacity_xs[:SAMPLES_PER_DATASET], udacity_ys[:SAMPLES_PER_DATASET]
    combined_xs = udacity_xs + dave_xs
    combined_ys = udacity_ys + dave_ys
    c = list(zip(combined_xs, combined_ys))
    random.shuffle(c)
    output_file = "udacity_dave.txt"
    with open(output_file, "w") as f:
        for img_path, label in c:
            f.write(f"{img_path}, {label}\n")

def create_udacity_label():
    import csv
    import random
    import os

    # === Configuration ===
    input_dir = os.path.join(basedir, 'driving/testing/')

    csv_input = input_dir + 'CH2_final_evaluation.csv'  # Input CSV path
    csv_output = 'udacity_label_shifted.csv'  # Output CSV path
    img_folder = input_dir + 'center/'  # Folder prefix for images if needed

    # === Load original data ===
    xs, ys = [], []
    with open(csv_input, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            xs.append(line.split(',')[0])
            ys.append(float(line.split(',')[1]))

    # === Categorize data ===
    data = list(zip(xs, ys))
    right = [d for d in data if d[1] < -0.2]
    left = [d for d in data if d[1] > 0.2]
    center = [d for d in data if abs(d[1]) <= 0.2]

    # === Define sampling targets ===
    n_total = len(ys)
    n_left = n_right = int(0.4 * n_total)
    n_center = n_total - n_left - n_right

    # === Random sampling ===
    random.seed(42)
    selected_left = random.choices(left, k=n_left) if len(left) < n_left else random.sample(left, k=n_left)
    selected_right = random.choices(right, k=n_right) if len(right) < n_right else random.sample(right, k=n_right)
    selected_center = random.choices(center, k=n_center) if len(center) < n_center else random.sample(center,
                                                                                                      k=n_center)

    # === Combine and shuffle ===
    final_data = selected_left + selected_right + selected_center
    random.shuffle(final_data)

    # === Write output CSV ===
    with open(csv_output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_id', 'steering_angle'])  # Header row
        for img_path, angle in final_data:
            writer.writerow([img_path, angle])

    print(f"Created '{csv_output}' with {len(final_data)} samples:")
    print(f"  - Left turns  (< -0.2): {sum(y < -0.2 for _, y in final_data)}")
    print(f"  - Right turns (>  0.2): {sum(y > 0.2 for _, y in final_data)}")
    print(f"  - Centered    (<= 0.2): {sum(abs(y) <= 0.2 for _, y in final_data)}")

def create_adv_udacity():
    epsilon = 8 / 255
    alpha = epsilon / 10  # Step size for iterative attacks
    TOTAL_SAMPLES = 5614
    SAMPLES_PER_DATASET = TOTAL_SAMPLES // 2  # 2807 from each

    model = keras.models.load_model(os.path.join(basedir, 'driving/epoch/epoch.h5')) #we attack lenet5

    # load udacity testing data
    input_dir = os.path.join(basedir, 'driving/testing/')
    csv_input = input_dir + 'CH2_final_evaluation.csv'  # Input CSV path
    img_folder = input_dir + 'center/'  # Folder prefix for images if needed

    # === Load original data ===
    udacity_xs, udacity_ys = [], []
    with open(csv_input, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            udacity_xs.append(img_folder + line.split(',')[0] + '.jpg')
            udacity_ys.append(float(line.split(',')[1]))

    # Select 5,000 random indices for adversarial examples
    np.random.seed(123)  # For reproducibility
    adv_indices = np.random.choice(len(udacity_ys), size=SAMPLES_PER_DATASET, replace=False)

    # Determine clean indices (excluding adversarial ones)
    all_indices = set(range(len(udacity_xs)))
    clean_indices = np.array(list(all_indices - set(adv_indices)))
    np.save('./clean_indices.npy', clean_indices)

    # Split indices for each attack
    split_size = SAMPLES_PER_DATASET // 3
    fgsm_indices = adv_indices[:split_size]
    bim_indices = adv_indices[split_size:2 * split_size]
    pgd_indices = adv_indices[2 * split_size:]

    # Generate adversarial examples
    X_fgsm_adv = fgsm_driving_attack(model, np.array(udacity_xs)[fgsm_indices], np.array(udacity_ys)[fgsm_indices], epsilon)
    np.save('./X_fgsm_adv.npy', X_fgsm_adv)
    np.save('./y_fgsm_adv.npy', np.array(udacity_ys)[fgsm_indices])
    X_bim_adv = bim_driving_attack(model, np.array(udacity_xs)[bim_indices], np.array(udacity_ys)[bim_indices], epsilon, alpha, iterations)
    np.save('./y_bim_adv.npy', np.array(udacity_ys)[bim_indices])
    np.save('./X_bim_adv.npy', X_bim_adv)
    X_pgd_adv = pgd_driving_attack(model, np.array(udacity_xs)[pgd_indices], np.array(udacity_ys)[pgd_indices], epsilon, alpha, iterations)
    np.save('./X_pgd_adv.npy', X_pgd_adv)
    np.save('./y_pgd_adv.npy', np.array(udacity_ys)[pgd_indices])


def create_adv_clean_file_udacity():
    # load udacity testing data
    input_dir = os.path.join(basedir, 'driving/testing/')
    csv_input = input_dir + 'CH2_final_evaluation.csv'  # Input CSV path
    img_folder = input_dir + 'center/'  # Folder prefix for images if needed

    # === Load original data ===
    udacity_xs, udacity_ys = [], []
    with open(csv_input, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            udacity_xs.append(img_folder + line.split(',')[0] + '.jpg')
            udacity_ys.append(float(line.split(',')[1]))

    # Extract clean images and labels
    clean_indices = np.load('./clean_indices.npy')
    X_clean_paths = np.array(udacity_xs)[clean_indices]
    X_clean = np.array([image.img_to_array(image.load_img(path, target_size=(100, 100))) for path in X_clean_paths])
    y_clean = np.array(udacity_ys)[clean_indices]

    X_fgsm_adv, y_fgsm_adv = np.load('./X_fgsm_adv.npy'), np.load('./y_fgsm_adv.npy')
    X_bim_adv, y_bim_adv = np.load('./X_bim_adv.npy'), np.load('./y_bim_adv.npy')
    X_pgd_adv, y_pgd_adv = np.load('./X_pgd_adv.npy'), np.load('./y_pgd_adv.npy')

    # Combine adversarial and clean examples
    X_combined = np.concatenate([X_fgsm_adv, X_bim_adv, X_pgd_adv, X_clean], axis=0)
    y_combined = np.concatenate([y_fgsm_adv, y_bim_adv, y_pgd_adv, y_clean], axis=0)

    combined_indices = np.arange(len(X_combined))
    np.random.shuffle(combined_indices)
    X_combined = X_combined[combined_indices]
    y_combined = y_combined[combined_indices]
    import pdb; pdb.set_trace()
    np.save(os.path.join(basedir, 'driving/data/fgsm_bim_pgd_clean_udacity_eps8_image.npy'), X_combined)
    np.save(os.path.join(basedir, 'driving/data/fgsm_bim_pgd_clean_udacity_eps8_label.npy'), y_combined)

if __name__ == '__main__':
    create_adv_mnist()