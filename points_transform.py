import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF


def preprocess_image_with_tracking(image_path, mode="crop"):
    """
    预处理图像并记录变换参数，用于坐标转换

    参数:
        image_path: 图像文件路径
        mode: 预处理模式，"crop" 或 "pad"

    返回:
        preprocessed_tensor: 预处理后的图像张量，形状 (1, 3, H, W)
        transform_info: 包含变换参数的字典
    """
    # 初始化变换信息
    transform_info = {
        'original_size': None,  # (width, height)
        'resized_size': None,  # (width, height) 调整大小后
        'final_size': None,  # (height, width) 最终张量尺寸
        'crop_params': None,  # (top, height) 裁剪参数，如果没有裁剪则为None
        'pad_params': None,  # (top, bottom, left, right) 填充参数，如果没有填充则为None
        'mode': mode,
        'target_size': 518
    }

    # 打开图像
    img = Image.open(image_path)

    # 记录原始尺寸
    orig_width, orig_height = img.size
    transform_info['original_size'] = (orig_width, orig_height)

    # 处理RGBA通道
    if img.mode == "RGBA":
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)

    # 转换为RGB
    img = img.convert("RGB")

    # 根据模式调整尺寸
    target_size = transform_info['target_size']

    if mode == "pad":
        # 使最大边长为518px，保持宽高比
        if orig_width >= orig_height:
            new_width = target_size
            new_height = round(orig_height * (new_width / orig_width) / 14) * 14
        else:
            new_height = target_size
            new_width = round(orig_width * (new_height / orig_height) / 14) * 14
    else:  # mode == "crop"
        new_width = target_size
        new_height = round(orig_height * (new_width / orig_width) / 14) * 14

    transform_info['resized_size'] = (new_width, new_height)

    # 调整图像大小
    img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
    img_tensor = TF.to_tensor(img)  # 转换为张量，形状 (3, H, W)

    # 裁剪处理（仅在crop模式且高度大于518时）
    if mode == "crop" and new_height > target_size:
        start_y = (new_height - target_size) // 2
        img_tensor = img_tensor[:, start_y:start_y + target_size, :]
        transform_info['crop_params'] = (start_y, target_size)
        final_height = target_size
    else:
        final_height = new_height

    final_width = new_width
    transform_info['final_size'] = (final_height, final_width)

    # 填充处理（仅在pad模式）
    if mode == "pad":
        h_padding = target_size - img_tensor.shape[1]
        w_padding = target_size - img_tensor.shape[2]

        if h_padding > 0 or w_padding > 0:
            pad_top = h_padding // 2
            pad_bottom = h_padding - pad_top
            pad_left = w_padding // 2
            pad_right = w_padding - pad_left

            transform_info['pad_params'] = (pad_top, pad_bottom, pad_left, pad_right)

            # 用白色填充（值=1.0）
            img_tensor = torch.nn.functional.pad(
                img_tensor, (pad_left, pad_right, pad_top, pad_bottom),
                mode="constant", value=1.0
            )

            # 更新最终尺寸
            transform_info['final_size'] = (target_size, target_size)

    # 添加批次维度
    img_tensor = img_tensor.unsqueeze(0)  # (1, 3, H, W)

    return img_tensor, transform_info


def original_to_preprocessed(point_xy, transform_info):
    """
    将原图坐标转换到预处理后图像的坐标

    参数:
        point_xy: 原图中的坐标，格式为 (x, y) 或 [[x1, y1], [x2, y2], ...]
        transform_info: 从 preprocess_image_with_tracking 返回的变换信息

    返回:
        预处理后图像中的坐标
    """
    # 确保输入为numpy数组
    points = np.array(point_xy, dtype=np.float32)
    if points.ndim == 1:
        points = points.reshape(1, -1)

    # 获取变换参数
    orig_width, orig_height = transform_info['original_size']
    resized_width, resized_height = transform_info['resized_size']

    # 1. 缩放转换
    scale_x = resized_width / orig_width
    scale_y = resized_height / orig_height
    transformed_points = points.copy()
    transformed_points[:, 0] *= scale_x
    transformed_points[:, 1] *= scale_y

    # 2. 裁剪转换（如果应用了裁剪）
    if transform_info['crop_params'] is not None:
        crop_top, crop_height = transform_info['crop_params']
        # 裁剪只是改变了y坐标的起点
        transformed_points[:, 1] -= crop_top

        # 检查点是否在裁剪区域内
        valid_mask = (transformed_points[:, 1] >= 0) & (transformed_points[:, 1] < crop_height)
        if not np.all(valid_mask):
            print(f"警告: {np.sum(~valid_mask)} 个点位于裁剪区域外")

    # 3. 填充转换（如果应用了填充）
    if transform_info['pad_params'] is not None:
        pad_top, pad_bottom, pad_left, pad_right = transform_info['pad_params']
        transformed_points[:, 0] += pad_left
        transformed_points[:, 1] += pad_top

    # 检查点是否在最终图像范围内
    final_height, final_width = transform_info['final_size']
    in_bounds_mask = (
            (transformed_points[:, 0] >= 0) &
            (transformed_points[:, 0] < final_width) &
            (transformed_points[:, 1] >= 0) &
            (transformed_points[:, 1] < final_height)
    )

    if not np.all(in_bounds_mask):
        print(f"警告: {np.sum(~in_bounds_mask)} 个点位于最终图像范围外")

    return transformed_points.squeeze()


def preprocessed_to_original(point_xy, transform_info):
    """
    将预处理后图像的坐标转换回原图坐标

    参数:
        point_xy: 预处理后图像中的坐标，格式为 (x, y) 或 [[x1, y1], [x2, y2], ...]
        transform_info: 从 preprocess_image_with_tracking 返回的变换信息

    返回:
        原图中的坐标
    """
    # 确保输入为numpy数组
    points = np.array(point_xy, dtype=np.float32)
    if points.ndim == 1:
        points = points.reshape(1, -1)

    # 获取变换参数
    orig_width, orig_height = transform_info['original_size']
    resized_width, resized_height = transform_info['resized_size']
    final_height, final_width = transform_info['final_size']

    # 创建反向转换的点
    transformed_points = points.copy()

    # 1. 反向填充转换（如果应用了填充）
    if transform_info['pad_params'] is not None:
        pad_top, pad_bottom, pad_left, pad_right = transform_info['pad_params']
        transformed_points[:, 0] -= pad_left
        transformed_points[:, 1] -= pad_top

    # 2. 反向裁剪转换（如果应用了裁剪）
    if transform_info['crop_params'] is not None:
        crop_top, crop_height = transform_info['crop_params']
        transformed_points[:, 1] += crop_top

    # 3. 反向缩放转换
    scale_x = resized_width / orig_width
    scale_y = resized_height / orig_height
    transformed_points[:, 0] /= scale_x
    transformed_points[:, 1] /= scale_y

    # 检查点是否在原图范围内
    in_bounds_mask = (
            (transformed_points[:, 0] >= 0) &
            (transformed_points[:, 0] < orig_width) &
            (transformed_points[:, 1] >= 0) &
            (transformed_points[:, 1] < orig_height)
    )

    if not np.all(in_bounds_mask):
        print(f"警告: {np.sum(~in_bounds_mask)} 个点位于原图范围外")

    return transformed_points.squeeze()


def batch_preprocess_with_tracking(image_path_list, mode="crop"):
    """
    批量预处理图像并记录每张图像的变换参数

    参数:
        image_path_list: 图像文件路径列表
        mode: 预处理模式，"crop" 或 "pad"

    返回:
        batch_tensor: 批量预处理后的图像张量，形状 (N, 3, H, W)
        transform_infos: 每张图像的变换信息列表
    """
    batch_tensors = []
    transform_infos = []

    for image_path in image_path_list:
        img_tensor, transform_info = preprocess_image_with_tracking(image_path, mode)
        batch_tensors.append(img_tensor)
        transform_infos.append(transform_info)

    # 堆叠所有张量
    batch_tensor = torch.cat(batch_tensors, dim=0)

    return batch_tensor, transform_infos


# 使用示例
def demonstrate_coordinate_transform():
    """演示坐标转换的完整流程"""

    # 1. 预处理图像并获取变换信息
    image_path = "examples/kitchen/images/00.png"
    mode = "crop"  # 或 "pad"

    preprocessed_tensor, transform_info = preprocess_image_with_tracking(image_path, mode)

    print(f"原始图像尺寸: {transform_info['original_size']}")
    print(f"调整后尺寸: {transform_info['resized_size']}")
    print(f"最终图像尺寸: {transform_info['final_size']}")
    print(f"模式: {transform_info['mode']}")

    if transform_info['crop_params']:
        print(f"裁剪参数: {transform_info['crop_params']}")
    if transform_info['pad_params']:
        print(f"填充参数: {transform_info['pad_params']}")

    # 2. 定义原图中的点（示例点）
    original_points = np.array([
        [100, 150],  # 左上区域
        [300, 200],  # 中心区域
        [500, 400],  # 右下区域
    ])

    print(f"\n原图点坐标: {original_points}")

    # 3. 正向转换：原图 -> 预处理后图像
    preprocessed_points = original_to_preprocessed(original_points, transform_info)
    print(f"预处理后图像点坐标: {preprocessed_points}")

    # 4. 反向转换：预处理后图像 -> 原图
    recovered_points = preprocessed_to_original(preprocessed_points, transform_info)
    print(f"恢复后的原图点坐标: {recovered_points}")

    # 5. 验证转换精度
    error = np.abs(recovered_points - original_points)
    print(f"\n转换误差 (x, y): {error}")
    print(f"最大误差: {error.max():.6f} 像素")

    # 可视化验证
    visualize_transform(original_points, preprocessed_points, recovered_points, transform_info)


def visualize_transform(original_pts, preprocessed_pts, recovered_pts, transform_info):
    """
    可视化坐标转换结果
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 绘制原图坐标空间
    axes[0].scatter(original_pts[:, 0], original_pts[:, 1], c='red', s=100, label='原图点')
    axes[0].scatter(recovered_pts[:, 0], recovered_pts[:, 1], c='blue', s=50, marker='x', label='恢复点')
    axes[0].set_xlim(0, transform_info['original_size'][0])
    axes[0].set_ylim(transform_info['original_size'][1], 0)  # 反转y轴，与图像坐标一致
    axes[0].set_title("原图坐标空间")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 绘制预处理后图像坐标空间
    final_height, final_width = transform_info['final_size']
    axes[1].scatter(preprocessed_pts[:, 0], preprocessed_pts[:, 1], c='green', s=100)
    axes[1].set_xlim(0, final_width)
    axes[1].set_ylim(final_height, 0)  # 反转y轴
    axes[1].set_title("预处理后图像坐标空间")
    axes[1].grid(True, alpha=0.3)

    # 绘制误差图
    error = np.abs(recovered_pts - original_pts)
    axes[2].bar(range(len(error)), error[:, 0], alpha=0.7, label='X方向误差')
    axes[2].bar(range(len(error)), error[:, 1], alpha=0.7, label='Y方向误差', bottom=error[:, 0])
    axes[2].set_xlabel("点索引")
    axes[2].set_ylabel("像素误差")
    axes[2].set_title("坐标转换误差")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()


# 批量处理示例
def batch_demo():
    """演示批量处理"""
    image_paths = [
        "examples/kitchen/images/00.png",
        "examples/kitchen/images/01.png",
        "examples/kitchen/images/02.png"
    ]

    # 批量预处理
    batch_tensor, transform_infos = batch_preprocess_with_tracking(image_paths, mode="crop")

    print(f"批量张量形状: {batch_tensor.shape}")

    # 为每张图像转换点坐标
    test_points = [np.array([[100, 150], [300, 200]]) for _ in range(len(image_paths))]

    for i, (points, info) in enumerate(zip(test_points, transform_infos)):
        print(f"\n图像 {i}:")
        preprocessed = original_to_preprocessed(points, info)
        recovered = preprocessed_to_original(preprocessed, info)
        print(f"  原图点: {points}")
        print(f"  预处理后点: {preprocessed}")
        print(f"  恢复点: {recovered}")
        print(f"  误差: {np.abs(recovered - points).max():.6f}")




"""
if __name__ == "__main__":
    # 运行演示
    demonstrate_coordinate_transform()

    # 批量处理演示
    # batch_demo()
"""