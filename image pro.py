import csv
import math

import cv2
import numpy as np
from PIL import Image

class QuadTreeNode:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = ()
        self.children0 = None
        self.children1 = None
        self.children2 = None
        self.children3 = None

class LinkedList:
    def __init__(self):
        self.head = None


class QuadTree:
    def __init__(self, image_array):
        self.image = image_array
        self.root = self.build_tree(0, 0, len(self.image), len(self.image[0]))

    def build_tree(self, x, y, width, height):
        if width <= 1 or height <= 1 or self.is_uniform(x, y, width, height):
            node = QuadTreeNode(x, y, width, height)
            node.color = self.average_color(x, y, width, height)
            return node

        node = QuadTreeNode(x, y, width, height)
        half_width = width // 2
        half_height = height // 2

        node.children0 = self.build_tree(x, y, half_width, half_height)
        node.children1 = self.build_tree(x + half_width, y, half_width, half_height)
        node.children2 = self.build_tree(x, y + half_height, half_width, half_height)
        node.children3 = self.build_tree(x + half_width, y + half_height, half_width, half_height)

        return node

    def is_uniform(self, x, y, width, height):
        first_color = self.image[x][y]
        for i in range(x, x + width):
            for j in range(y, y + height):
                if self.image[i][j] != first_color:
                    return False
        return True

    def average_color(self, x, y, width, height):
        total_color_r = 0
        total_color_g = 0
        total_color_b = 0
        count = 0
        for i in range(x, x + width):
            for j in range(y, y + height):
                total_color_r += self.image[i][j][0]
                total_color_g += self.image[i][j][1]
                total_color_b += self.image[i][j][2]
                count += 1

        return (total_color_r // count, total_color_g // count, total_color_b // count) if count > 0 else (0, 0, 0)

    def tree_depth(self):
        return self.get_depth(self.root)

    def get_depth(self, node):
        if node.children0 == None and node.children1 == None and node.children2 == None and node.children3 == None:
            return 1
        return 1 + max(self.get_depth(node.children0), self.get_depth(node.children1), self.get_depth(node.children2),
                       self.get_depth(node.children3))

    def pixel_depth(self, px, py):
        return self.get_pixel_depth(self.root, px, py, 0)

    def get_pixel_depth(self, node, px, py, depth):
        if node is None:
            return -1
        if px < node.x or px >= node.x + node.width or py < node.y or py >= node.y + node.height:
            return -1
        if node.children0 == None and node.children1 == None and node.children2 == None and node.children3 == None:
            return depth

        depths = []
        depths.append(self.get_pixel_depth(node.children0, px, py, depth + 1))
        depths.append(self.get_pixel_depth(node.children1, px, py, depth + 1))
        depths.append(self.get_pixel_depth(node.children2, px, py, depth + 1))
        depths.append(self.get_pixel_depth(node.children3, px, py, depth + 1))

        return max(depths)

    def search_subspaces_with_range(self, x1, y1, x2, y2):
        result_image = self._search_subspaces_with_range(self.root, x1, y1, x2, y2)
        return result_image

    def _search_subspaces_with_range(self, node, x1, y1, x2, y2):
        if node is None or not self.overlaps(node, x1, y1, x2, y2):
            return None

        if node.children0 == None and node.children1 == None and node.children2 == None and node.children3 == None:
            sub_node = QuadTreeNode(node.x, node.y, node.width, node.height)
            sub_node.color = node.color
            return sub_node

        sub_node = QuadTreeNode(node.x, node.y, node.width, node.height)
        sub_node.children0 = self._search_subspaces_with_range(node.children0, x1, y1, x2, y2)
        sub_node.children1 = self._search_subspaces_with_range(node.children1, x1, y1, x2, y2)
        sub_node.children2 = self._search_subspaces_with_range(node.children2, x1, y1, x2, y2)
        sub_node.children3 = self._search_subspaces_with_range(node.children3, x1, y1, x2, y2)

        return sub_node

    def overlaps(self, node, x1, y1, x2, y2):
        return not (x2 < node.x or x1 > node.x + node.width or y2 < node.y or y1 > node.y + node.height)

    def mask(self, x1, y1, x2, y2):
        masked_image = self._mask(self.root, x1, y1, x2, y2)
        return masked_image

    def _mask(self, node, x1, y1, x2, y2):
        if node is None or not self.overlaps(node, x1, y1, x2, y2):
            return None

        if node.children0 == None and node.children1 == None and node.children2 == None and node.children3 == None:
            mask_node = QuadTreeNode(node.x, node.y, node.width, node.height)
            mask_node.color = (0, 0, 0)
            return mask_node

        mask_node = QuadTreeNode(node.x, node.y, node.width, node.height)
        mask_node.children0 = self._mask(node.children0, x1, y1, x2, y2)
        mask_node.children1 = self._mask(node.children1, x1, y1, x2, y2)
        mask_node.children2 = self._mask(node.children2, x1, y1, x2, y2)
        mask_node.children3 = self._mask(node.children3, x1, y1, x2, y2)

        return mask_node

    def array_to_image(self, array):
        height = len(array)
        width = len(array[0]) if height > 0 else 0

        img = Image.new("RGB", (width, height))

        for y in range(height):
            for x in range(width):
                if isinstance(array[y][x], tuple) and len(array[y][x]) == 3:
                    img.putpixel((x, y), array[y][x])
                else:
                    img.putpixel((x, y), (0, 0, 0))

        return img

    def compress(self, max_depth):
        compressed_tree_root = self.compress_tree(self.root, max_depth, 0)
        return compressed_tree_root

    def compress_tree(self, node, max_depth, current_depth):
        if node is None:
            return None

        if current_depth == max_depth or (node.children0 is None and node.children1 is None and node.children2 is None and node.children3 is None):
            node.color = self.average_color(node.x, node.y, node.width, node.height)
            compressed_node = QuadTreeNode(node.x, node.y, node.width, node.height)
            compressed_node.color = node.color
            return compressed_node

        compressed_node = QuadTreeNode(node.x, node.y, node.width, node.height)
        compressed_node.children0 = self.compress_tree(node.children0, max_depth, current_depth + 1)
        compressed_node.children1 = self.compress_tree(node.children1, max_depth, current_depth + 1)
        compressed_node.children2 = self.compress_tree(node.children2, max_depth, current_depth + 1)
        compressed_node.children3 = self.compress_tree(node.children3, max_depth, current_depth + 1)

        return compressed_node

def image_to_array(file_path):
    img = Image.open(file_path)
    img = img.convert('RGB')

    image_array = []
    for y in range(img.height):
        row = []
        for x in range(img.width):
            r, g, b = img.getpixel((x, y))
            row.append((r, g, b))
        image_array.append(row)

    return image_array


def process_video(video_path, output_video_path1, output_video_path2, output_video_path3, x1, y1, x2, y2, max_depth):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out1 = cv2.VideoWriter(output_video_path1, fourcc, 30.0, (frame_width, frame_height))
    out2 = cv2.VideoWriter(output_video_path2, fourcc, 30.0, (frame_width, frame_height))
    out3 = cv2.VideoWriter(output_video_path3, fourcc, 30.0, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_array = [list(map(tuple, row)) for row in rgb_frame.tolist()]

        quad_tree = QuadTree(image_array)

        masked_frame = quad_tree.mask(x1, y1, x2, y2)
        masked_frame2 = [row[:] for row in quad_tree.image]
        show_compressed_tree(masked_frame, masked_frame2)
        masked_frame1 = [list(map(list, row)) for row in masked_frame2]
        masked_frame1 = np.array(masked_frame1, dtype=np.uint8)

        search_frame = quad_tree.search_subspaces_with_range(x1, y1, x2, y2)
        search_frame2 = [[(255, 255, 255) for _ in range(len(quad_tree.image[0]))] for _ in range(len(quad_tree.image))]
        show_compressed_tree(search_frame, search_frame2)
        search_frame1 = [list(map(list, row)) for row in search_frame2]
        search_frame1 = np.array(search_frame1, dtype=np.uint8)

        compress_frame = quad_tree.compress(max_depth)
        compress_frame2 = [[(0, 0, 0) for _ in range(len(quad_tree.image[0]))] for _ in range(len(quad_tree.image))]
        show_compressed_tree(compress_frame, compress_frame2)
        compress_frame1 = [list(map(list, row)) for row in compress_frame2 if all(item is not None for item in row)]
        compress_frame1 = np.array(compress_frame1, dtype=np.uint8)

        out_frame1 = cv2.cvtColor(masked_frame1.astype(np.uint8), cv2.COLOR_RGB2BGR)
        out_frame2 = cv2.cvtColor(search_frame1.astype(np.uint8), cv2.COLOR_RGB2BGR)
        out_frame3 = cv2.cvtColor(compress_frame1.astype(np.uint8), cv2.COLOR_RGB2BGR)

        out1.write(out_frame1)
        out2.write(out_frame2)
        out3.write(out_frame3)

    cap.release()
    out1.release()
    out2.release()
    out3.release()


def show_compressed_tree(node, image):
    if node is None:
        return

    if node.children0 is None and node.children1 is None and node.children2 is None and node.children3 is None:
        for i in range(node.x, node.x + node.width):
            for j in range(node.y, node.y + node.height):
                if 0 <= i < len(image) and 0 <= j < len(image[0]):
                    image[i][j] = node.color
        return

    show_compressed_tree(node.children0, image)
    show_compressed_tree(node.children1, image)
    show_compressed_tree(node.children2, image)
    show_compressed_tree(node.children3, image)


def array_to_imagee(array):
    height = len(array)
    width = len(array[0]) if height > 0 else 0

    img = Image.new("RGB", (width, height))

    for y in range(height):
        for x in range(width):
            if isinstance(array[y][x], tuple) and len(array[y][x]) == 3:
                img.putpixel((x, y), array[y][x])
            else:
                img.putpixel((x, y), (0, 0, 0))

    return img

def read_file(path):
    with open(path) as file:
        file.readline()
        result = []
        for item in next(csv.reader(file)):
            values = list(map(int, item.split(",")))
            if len(values) == 1:
                gray = values[0]
                result.append((gray, gray, gray))
            elif len(values) == 3:
                r, g, b = reversed(values)
                result.append((r, g, b))
    length = int(math.sqrt(len(result)))
    return [result[i : i + length] for i in range(0, length * length, length)]


def main():
    image = image_to_array(r"Photo\test.jpg")
    #image = read_file(r"Photo\image1_gray.csv")
    quad_tree = QuadTree(image)

    depth = quad_tree.tree_depth()
    print("Tree Depth:", depth)

    pixel_depth = quad_tree.pixel_depth(420, 450)
    print("Pixel Depth:", pixel_depth)

    space_tree = quad_tree.search_subspaces_with_range(1, 1, 480, 480)
    new_image_array = [[(255, 255, 255) for _ in range(len(quad_tree.image[0]))] for _ in range(len(quad_tree.image))]
    show_compressed_tree(space_tree, new_image_array)
    new_image = array_to_imagee(new_image_array)
    new_image.save(r"Photo\search_image.jpg")

    mask_tree = quad_tree.mask(1, 1, 480, 480)
    masked_image_array = [row[:] for row in quad_tree.image]
    show_compressed_tree(mask_tree, masked_image_array)
    masked_image = array_to_imagee(masked_image_array)
    masked_image.save(r"Photo\masked_image.jpg")

    compressed_tree = quad_tree.compress(7)
    visualized_image = [[(0, 0, 0) for _ in range(len(image[0]))] for _ in range(len(image))]
    show_compressed_tree(compressed_tree, visualized_image)
    visualized_pil_image = array_to_imagee(visualized_image)
    visualized_pil_image.save(r"Photo\compress_image.jpg")

    process_video(r"Photo\ftest.mp4", r"Photo\ftest1.avi", r"Photo\ftest2.avi", r"Photo\ftest3.avi", 2, 2, 480, 480, 5)


if __name__ == "__main__":
    main()