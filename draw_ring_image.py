import numpy as np
import matplotlib.pyplot as plt


def draw_single_circle(width, height, inner_radius, outer_radius):
    # 创建空白图像
    image = np.zeros((height, width))

    # 计算图像中心
    center_x = width // 2
    center_y = height // 2

    # 绘制圆环
    for y in range(height):
        for x in range(width):
            distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            if inner_radius <= distance <= outer_radius:
                image[y, x] = 1

    # 生成带参数的图像文件名
    single_circle_filename = f"{inner_radius}-{outer_radius}-single_circle_image.png"
    # 保存图像
    plt.imsave(single_circle_filename, image, cmap='gray')

    # 使用 Matplotlib 展示图像
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.title('Single Circle Image')
    plt.show()


def draw_hexagonal_circle_lattice(width, height, inner_radius, outer_radius, lattice_length):
    # 创建空白图像
    image = np.zeros((height, width))

    # 计算六边形晶格的偏移量
    dx = lattice_length
    dy = np.sqrt(3) / 2 * lattice_length

    # 计算可以容纳的晶格列数和行数
    num_cols = width // dx
    num_rows = height // int(dy)

    # 绘制六边形晶格分布的圆环点阵
    for row in range(num_rows):
        for col in range(num_cols):
            # 计算当前点的中心坐标
            if row % 2 == 0:
                center_x = col * dx
            else:
                center_x = col * dx + dx // 2
            center_y = row * int(dy)

            # 确保点在图像范围内
            if 0 <= center_x < width and 0 <= center_y < height:
                # 绘制圆环
                for y in range(max(0, center_y - outer_radius), min(height, center_y + outer_radius + 1)):
                    for x in range(max(0, center_x - outer_radius), min(width, center_x + outer_radius + 1)):
                        distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                        if inner_radius <= distance <= outer_radius:
                            image[y, x] = 1

    # 生成带参数的图像文件名
    lattice_filename = f"{inner_radius}-{outer_radius}-{lattice_length}-hexagonal_circle_lattice.png"
    # 保存图像
    plt.imsave(lattice_filename, image, cmap='gray')

    # 使用 Matplotlib 展示图像
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.title('Hexagonal Circle Lattice Image')
    plt.show()


if __name__ == "__main__":
    """
    生成单个圆环和六边形晶格分布的圆环点阵图像
    """

    # 图像尺寸
    width, height = 1920, 1080
    # 可修改的圆环内外半径
    inner_radius = 10
    outer_radius = 20
    # 可修改的点阵间隔
    lattice_length = 100

    draw_single_circle(width, height, inner_radius, outer_radius)
    draw_hexagonal_circle_lattice(width, height, inner_radius, outer_radius, lattice_length)
