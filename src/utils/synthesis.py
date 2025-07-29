import math
import tensorflow as tf
import numpy as np
import flare.utils as utils

tf.experimental.numpy.experimental_enable_numpy_behavior()

#TODO tone mapping 和 gamma矫正的顺序好像反了？
def mixup(scene, flare,mode='ISP',gamma=2):
  # I = flare[:,:,0]+flare[:,:,1]+flare[:,:,2]
  # I = I / 3
  #change by hhq, intensity estimation based on quantum efficiency
  I = flare[:,:,0]*0.2126+flare[:,:,1]*0.7152+flare[:,:,2]*0.0722

  # a = np.random.random()*4+3
  # 黑洞原因：基于凸组合，当flare很小时强行拉大，此时flare强度远低于场景，导致flare处明显变暗
  # 关键是，它的强度分配只基于flare，没考虑flare逐渐变暗消失时，已经不满足它的flare强度假设
  # 因此必须和场景共同考虑，建立一个当flare逐渐消失时依然成立的凸组合
  # 这需要从头推翻它的方法？
  # why sigmoid? is there any better way?它的证明只涉及两个极端，中间是一概没提，有点正确的废话
  if mode == 'ISP':
    #TODO,怎么改？
    #不tf.reduce_max(tf.compat.v1.layers.flatten(weight))，是否可行？
    a = 2
    # weight = 1/(1+np.e**(-a*(I-0.5)))
    # weight = np.sqrt(I)
    weight = I**3 + 2 * I*(1-I)
    # k = 10
    # weight = 0.5 * (1 + np.tanh(k * (I - 0.5)))
    # weight = weight - tf.reduce_min(tf.compat.v1.layers.flatten(weight))
    # weight = weight/tf.reduce_max(tf.compat.v1.layers.flatten(weight))
    a1 = (scene[:,:,0]*(1-weight)+flare[:,:,0]*weight)
    a2 = (scene[:,:,1]*(1-weight)+flare[:,:,1]*weight)
    a3 = (scene[:,:,2]*(1-weight)+flare[:,:,2]*weight)
    scene_rgb = tf.stack([scene[:, :, 0]*(1-weight),scene[:, :, 1]*(1-weight),scene[:, :, 2]*(1-weight)], axis=-1)
    flare_rgb = tf.stack([flare[:, :, 0]*weight,flare[:, :, 1]*weight,flare[:, :, 2]*weight], axis=-1)
    return tf.clip_by_value(tf.stack([a1,a2,a3], axis=-1), 0.0, 1.0),scene_rgb,flare_rgb
  elif mode == 'analytic' or mode == 'ACES':
    # 定义运算函数
    def transform(x):
        # 第一步计算 1/2 + cos(1/3 * (arccos(2x - 1) + pi))
        # intermediate = 1/2 + np.cos(1/3 * (np.arccos(2 * x - 1) + np.pi))
        # changed, use sin instead of cos
        intermediate = 1/2 - np.sin(1/3 * (np.arcsin(1-2*x)))
        return intermediate
    def final_transform(x):
        # 第二步计算 3x^2 - 2x^3
        return 3 * x**2 - 2 * x**3
    # Forward function: ACES tone mapping
    def ACES(color):
        A = 2.51
        B = 0.03
        C = 2.43
        D = 0.59
        E = 0.14
        return (color * (A * color + B)) / (color * (C * color + D) + E)

    # Inverse function: Solve the quadratic equation to get the reverse mapping
    def ACES_reverse(x):
        A = 2.51
        B = 0.03
        C = 2.43
        D = 0.59
        E = 0.14

        # Coefficients of the quadratic equation
        a = x * C - A
        b = x * D - B
        c = x * E

        # Calculate the discriminant
        discriminant = b**2 - 4 * a * c

        # Check for valid solutions
        if np.any(discriminant < 0):
            raise ValueError("No real solution exists for some input values.")

        # Solve the quadratic equation using the positive root
        root = (-b - np.sqrt(discriminant)) / (2 * a)
        return root
    def ACES_profession(x):
      # 定义输入和输出的转换矩阵
      ACESInputMat = np.array([
          [0.59719, 0.35458, 0.04823],
          [0.07600, 0.90834, 0.01566],
          [0.02840, 0.13383, 0.83777]
      ])

      ACESOutputMat = np.array([
          [1.60475, -0.53108, -0.07367],
          [-0.10208, 1.10813, -0.00605],
          [-0.00327, -0.07276, 1.07602]
      ])

      def RRTAndODTFit(v):
        """
        模拟 HLSL 的 RRTAndODTFit 函数
        """
        a = v * (v + 0.0245786) - 0.000090537
        b = v * (0.983729 * v + 0.4329510) + 0.238081
        return a / b

      # 将图像展平为二维矩阵 (N, 3)，其中 N 是像素数
      original_shape = x.shape
      color = x.reshape(-1, 3).T

      # 转换为线性空间
      color = np.dot(ACESInputMat, color)

      # 应用 RRT 和 ODT 映射
      color = RRTAndODTFit(color)

      # 转换为 sRGB 空间
      color = np.dot(ACESOutputMat, color)

      # 恢复为原始图像形状
      color = color.T.reshape(original_shape)

      return color

    def ACES_profession_reverse(x):
      # 定义输入和输出的转换矩阵
      ACESInputMat = np.array([
          [0.59719, 0.35458, 0.04823],
          [0.07600, 0.90834, 0.01566],
          [0.02840, 0.13383, 0.83777]
      ])

      ACESOutputMat = np.array([
          [1.60475, -0.53108, -0.07367],
          [-0.10208, 1.10813, -0.00605],
          [-0.00327, -0.07276, 1.07602]
      ])

      ACESInputMat_inv = np.linalg.inv(ACESInputMat)
      ACESOutputMat_inv = np.linalg.inv(ACESOutputMat)

      def RRTAndODTFitInverse(y):
        """
        计算 RRTAndODTFit 的逆函数
        """
        A = 0.983729 * y - 1
        B = 0.4329510 * y - 0.0245786
        C = 0.238081 * y + 0.000090537

        discriminant = B**2 - 4 * A * C
        sqrt_discriminant = np.sqrt(discriminant)

        # 选择符合 v > 0 的解
        v2 = (-B - sqrt_discriminant) / (2 * A)
        return v2

      # 将图像展平为二维矩阵 (N, 3)，其中 N 是像素数
      original_shape = x.shape
      color = x.reshape(-1, 3).T

      # 转换为线性空间
      color = np.dot(ACESOutputMat_inv, color)

      # 应用 RRT 和 ODT 映射的逆函数
      color = RRTAndODTFitInverse(color)

      # 转换为 sRGB 空间
      color = np.dot(ACESInputMat_inv, color)

      # 恢复为原始图像形状
      color = color.T.reshape(original_shape)

      return color
    if mode == "analytic":
      transformed_scene = transform(scene)
      transformed_flare = transform(flare)
      transformed_scene = tf.image.adjust_gamma(transformed_scene, gamma)
      transformed_flare = tf.image.adjust_gamma(transformed_flare, gamma)
      #to the highest
      #TODO, 这样做是因为，上边多项式描述一般图像的tone mapping,但是极端图像有更夸张的非线性调整，原函数体现不出来
      #这个函数同时完成高动态范围的取舍，与数字信号转模拟信号
      transformed_scene = transformed_scene/(1-transformed_scene+0.000000001)
      transformed_flare = transformed_flare/(1-transformed_flare+0.000000001)
    elif mode == "ACES":
      #似乎此时gamma矫正要在tone之后，不然范围对不上？这个需要理清楚
      transformed_scene = tf.image.adjust_gamma(scene, gamma)
      transformed_flare = tf.image.adjust_gamma(flare, gamma)
      # transformed_scene = ACES_reverse(transformed_scene)
      # transformed_flare = ACES_reverse(transformed_flare)
      # transformed_scene = ACES_reverse(scene)
      # transformed_flare = ACES_reverse(flare)

      transformed_scene = ACES_profession_reverse(transformed_scene)
      transformed_flare = ACES_profession_reverse(transformed_flare)



    # 加和两个 tensor,考虑增益，假设相机ISP增益相同
    gain_scene = 1
    gain_flare = 1.2
    gain_combine = gain_flare
    transformed_scene = transformed_scene * gain_scene
    transformed_flare = transformed_flare * gain_flare
    combined = (transformed_scene + transformed_flare)/gain_combine

    #back to 0-1
    if mode == "analytic":
      combined = combined/(1+combined)

      # 对加和结果应用第二步运算
      combined = tf.image.adjust_gamma(combined, 1.0/gamma)
      final_result = final_transform(combined)
    elif mode == "ACES":
      #似乎此时gamma矫正要在tone之后，不然范围对不上？这个需要理清楚
      # final_result = ACES(combined)
      final_result = ACES_profession(combined)
      final_result = tf.image.adjust_gamma(final_result, 1.0/gamma)

    return tf.clip_by_value(final_result, 0.0, 1.0),scene,flare


  else:
    a = 0
    weight = 1/(1+np.e**(-a*(I-0.5)))
    a1 = (scene[:,:,0]*(1-weight)+flare[:,:,0]*weight)*2
    a2 = (scene[:,:,1]*(1-weight)+flare[:,:,1]*weight)*2
    a3 = (scene[:,:,2]*(1-weight)+flare[:,:,2]*weight)*2
    return tf.clip_by_value(tf.stack([a1,a2,a3], axis=-1), 0.0, 1.0),scene,flare

def add_flare(scene,
              flare,
              noise,
              flare_max_gain = 10.0,
              apply_affine = True,
              rotation = 0.,
              shift=[0,100],
              shear=[0.0,0.0],
              scale=[1.0,1.0],
              mode='ISP',
              gamma=2,
              theta = 0):
  """Adds flare to natural images.
  Returns:
    - Flare-free scene in sRGB.
    - Flare-only image in sRGB.
    - Scene with flare in sRGB.
    - Gamma value used during synthesis.
  """
  # batch_size, flare_input_height, flare_input_width, _ = flare.shape

  # Since the gamma encoding is unknown, we use a random value so that the model
  # will hopefully generalize to a reasonable range of gammas.
  # gamma = tf.random.uniform([], 1.8, 2.2)
  # gamma = 2.0
  # convert to Raw? but no tone mapping 
  # 如果输入是一般RGB,这一整套都是反的？
  # 是的，伽马值 2.2 通常是指在图像信号处理（ISP）流程中，将 Raw 图像的线性强度值 转换为最终输出图像（如 sRGB 或 Rec.709 格式）中的 非线性强度值。这一步是为了匹配人眼对亮度的感知特性，最终显示在屏幕上的图像会更符合视觉效果。
  # 所以 ISP gamma 需要是一个 1/2.2 encode 的 gamma， gpt不可信。。。

  flare_linear = tf.image.adjust_gamma(flare, gamma)
  # flare_linear = flare_linear*np.cos(theta)

  # Remove DC background in flare.
  # flare_linear = utils.remove_background(flare_linear)

  if apply_affine:
    # rotation = tf.random.uniform([batch_size], minval=-math.pi, maxval=math.pi)
    # shift = tf.random.normal([batch_size, 2], mean=0.0, stddev=10.0)
    # shear = tf.random.uniform([batch_size, 2],
    #                           minval=-math.pi / 9,
    #                           maxval=math.pi / 9)
    # scale = tf.random.uniform([batch_size, 2], minval=0.7, maxval=1.1)

    flare_linear = utils.apply_affine_transform(
        flare_linear,
        rotation=rotation,
        shift_x=shift[0],
        shift_y=shift[1],
        shear_x=shear[0],
        shear_y=shear[1],
        scale_x=scale[0],
        scale_y=scale[1])

  flare_linear = tf.clip_by_value(flare_linear, 0.0, 1.0)

  # flare_linear = tf.image.random_flip_left_right(
  #     tf.image.random_flip_up_down(flare_linear))

  # First normalize the white balance. Then apply random white balance.

  # flare_linear = utils.normalize_white_balance(flare_linear)
  # rgb_gains = tf.random.uniform([3], 0, flare_max_gain, dtype=tf.float32)
  # flare_linear *= rgb_gains

  # Further augmentation on flare patterns: random blur and DC offset.
  # blur_size = tf.random.uniform([], 0.1, 3)
  # flare_linear = utils.apply_blur(flare_linear, blur_size)
  # offset = tf.random.uniform([], -0.02, 0.02)
  # flare_linear = tf.clip_by_value(flare_linear + offset, 0.0, 1.0)

  # flare_srgb = tf.image.adjust_gamma(flare_linear, 1.0 / gamma)
  # flare_srgb = flare_linear

  # Scene augmentation: random crop and flips.
  scene_linear = tf.image.adjust_gamma(scene, gamma)
  # scene_linear = tf.image.random_crop(scene_linear, flare_linear.shape)
  # scene_linear = tf.image.random_flip_left_right(
  #     tf.image.random_flip_up_down(scene_linear))

  # Additive Gaussian noise. The Gaussian's variance is drawn from a Chi-squared
  # distribution. This is equivalent to drawing the Gaussian's standard
  # deviation from a truncated normal distribution, as shown below.
  sigma = tf.abs(tf.random.normal([], 0, noise))
  noise = tf.random.normal(scene_linear.shape, 0, sigma)
  scene_linear += noise

  # Random digital gain.
  # TODO, hyperparameter, bg become dark due to autoexposure
  # gain = tf.random.uniform([], 0.85, 0.85)  # varying the intensity scale
  # scene_linear = tf.clip_by_value(gain * scene_linear, 0.0, 1.0)

  # scene_srgb = tf.image.adjust_gamma(scene_linear, 1.0 / gamma)
  # scene_srgb = scene_linear

  # Combine the flare-free scene with a flare pattern to produce a synthetic
  # training example.
  # combined_1 = mixup(scene_linear[0], flare_linear[0])
  # combined_2 = mixup(scene_linear[1], flare_linear[1])
  # combined_srgb = tf.stack([combined_1, combined_2])
  if mode=='analytic':
    combined_srgb,scene_srgb,flare_srgb = mixup(tf.image.adjust_gamma(scene_linear, 1.0 / gamma), tf.image.adjust_gamma(flare_linear, 1.0 / gamma),mode=mode,gamma=gamma)
  else:
    combined_srgb,scene_srgb,flare_srgb = mixup(scene_linear, flare_linear,mode=mode)
    combined_srgb = tf.image.adjust_gamma(combined_srgb, 1.0 / gamma)
    combined_srgb = tf.clip_by_value(combined_srgb, 0.0, 1.0)

    scene_srgb = tf.image.adjust_gamma(scene_srgb, 1.0 / gamma)
    scene_srgb = tf.clip_by_value(scene_srgb, 0.0, 1.0)

    flare_srgb = tf.image.adjust_gamma(flare_srgb, 1.0 / gamma)
    flare_srgb = tf.clip_by_value(flare_srgb, 0.0, 1.0)
  
  return (utils.quantize_8(scene_srgb), utils.quantize_8(flare_srgb),
          utils.quantize_8(combined_srgb), gamma)

def load_images(scene_path, flare_path):
    """Loads and processes scene and flare images for further use.

    Args:
      scene_path: Path to the scene image file.
      flare_path: Path to the flare image file.

    Returns:
      - scene: Preprocessed scene image tensor.
      - flare: Preprocessed flare image tensor, resized to match scene's size.
    """
    
    # Load the scene and flare images as tensors
    scene = tf.io.read_file(scene_path)
    flare = tf.io.read_file(flare_path)

    # Decode the images (assuming they are in standard formats like PNG/JPEG)
    scene = tf.image.decode_image(scene, channels=3)  # Scene should be 3-channel (RGB)
    flare = tf.image.decode_image(flare, channels=3)  # Flare should also be 3-channel (RGB)

    # Normalize the images to the range [0, 1]
    scene = tf.cast(scene, tf.float32) / 255.0
    flare = tf.cast(flare, tf.float32) / 255.0

    # Get the dimensions of the scene image
    scene_shape = tf.shape(scene)

    # Resize the flare image to match the scene image's dimensions
    flare = tf.image.resize(flare, [scene_shape[0], scene_shape[1]])

    return scene, flare
def save_image(image_tensor, output_path, file_format='jpeg'):
    """Saves a TensorFlow image tensor to disk in the specified format.
    
    Args:
        image_tensor: The image tensor to be saved.
        output_path: Path where the image will be saved.
        file_format: The format to save the image, either 'jpeg' or 'png' (default is 'jpeg').
    """
    # Ensure the image tensor is in the [0, 1] range
    image_tensor = tf.clip_by_value(image_tensor, 0.0, 1.0)

    # Convert the image tensor to 8-bit by multiplying by 255 and converting to uint8
    image_tensor = tf.cast(image_tensor * 255.0, tf.uint8)

    # Encode the image tensor to the desired format
    if file_format.lower() == 'jpeg':
        encoded_image = tf.image.encode_jpeg(image_tensor)
    elif file_format.lower() == 'png':
        encoded_image = tf.image.encode_png(image_tensor)
    else:
        raise ValueError("Unsupported file format. Use 'jpeg' or 'png'.")

    # Write the encoded image to the specified file
    tf.io.write_file(output_path, encoded_image)
def position_to_shift(position, image_size):
    """
    Convert position (with respect to top-left corner) to shift (with respect to image center).
    
    Args:
      position: A list [pos_x, pos_y] representing the center of the flare in the image.
      image_size: A tuple (W, H) representing the width and height of the image.
    
    Returns:
      shift: A list [shift_x, shift_y] representing the movement of the flare center from the image center.
    """
    W, H,_ = image_size
    
    # Calculate the center of the image
    center_x = W / 2
    center_y = H / 2
    
    # Convert position to shift
    shift_x = position[0] - center_x
    shift_y = position[1] - center_y
    
    return [shift_x, shift_y]
scene_path = "D:/2025/event_simu/flare/scene/12798.jpg"
flare_path = "D:/2025/event_simu/flare/Compound_Flare/000935.png"

scene, flare = load_images(scene_path, flare_path)
position = [100, 100]
def image_add_flare(scene, flare, position,mode='ISP',theta=0):
    image_size = tf.shape(scene)


    shift = position_to_shift(position, image_size)
    # 将场景和flare图像传递给 `add_flare_at_position` 函数
    scene_srgb, flare_srgb, scene_with_flare, gamma = add_flare(scene, flare, noise=0, shift=shift,mode=mode,theta=theta)
    return scene_srgb, flare_srgb, scene_with_flare, gamma
image_add_flare(scene, flare, position)
#TODO, single image change to video, add  point light source