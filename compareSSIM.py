import cv2
from skimage.metrics import structural_similarity as ssim
import numpy as np

def calculate_ssim(image1_path, image2_path):
    """
    计算两张图片之间的SSIM指数
    :param image1_path: 第一张图片的路径
    :param image2_path: 第二张图片的路径
    :return: SSIM数值（范围0-1，越接近1表示越相似）
    """
    # 读取图片
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    # 检查图片是否成功读取
    if img1 is None:
        raise ValueError(f"无法读取图像: {image1_path}")
    if img2 is None:
        raise ValueError(f"无法读取图像: {image2_path}")

    # 验证图片尺寸是否相同
    if img1.shape != img2.shape:
        raise ValueError("图片尺寸不一致，无法计算SSIM")

    # 转换颜色空间
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    try:
        return ssim(img1, img2, 
                   data_range=255, 
                   channel_axis=2,
                   win_size=3)  # 适当减小窗口尺寸加快计算
    except TypeError:
        # 回退到旧版本skimage的multichannel参数
        return ssim(img1, img2, 
                   data_range=255, 
                   multichannel=True,
                   win_size=3)

if __name__ == "__main__":
    
    original_path = "demo\GOPR0384_11_00-000001.png"    # 原图路径
    restored_path = "demo\Gopro_me_test.png"    # 去模糊图路径
    
    try:
        ssim_index = calculate_ssim(original_path, restored_path)
        print(f"SSIM指数: {ssim_index:.4f}")
        print("提示：SSIM范围0-1，数值越大表示越相似")
        print("一般判定标准：")
        print("   > 0.95：几乎无差异")
        print("0.90-0.95：微小差异")
        print("0.80-0.90：明显差异")
        print("   < 0.80：差异显著")
    except Exception as e:
        print(f"发生错误: {str(e)}")