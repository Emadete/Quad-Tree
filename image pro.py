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

        return (total_color_r // count, total_color_g // count, total_color_b // count) if count > 0 else 0

    def tree_depth(self):
        return self.get_depth(self.root)

    def get_depth(self, node):
        if node.children0 == None and node.children1 == None and node.children2 == None and node.children3 == None:
            return 1
        return 1 + max(self.get_depth(node.children0), self.get_depth(node.children1), self.get_depth(node.children2), self.get_depth(node.children3))

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
        result_image = [[(255, 255, 255) for _ in range(len(self.image[0]))] for _ in range(len(self.image))]

        self._search_subspaces_with_range(self.root, x1, y1, x2, y2, result_image)

        return result_image

    def _search_subspaces_with_range(self, node, x1, y1, x2, y2, result_image):
        if node is None or not self.overlaps(node, x1, y1, x2, y2):
            return

        if node.children0 == None and node.children1 == None and node.children2 == None and node.children3 == None:

            for i in range(node.x, node.x + node.width):
                for j in range(node.y, node.y + node.height):
                    result_image[i][j] = node.color
            return

        self._search_subspaces_with_range(node.children0, x1, y1, x2, y2, result_image)
        self._search_subspaces_with_range(node.children1, x1, y1, x2, y2, result_image)
        self._search_subspaces_with_range(node.children2, x1, y1, x2, y2, result_image)
        self._search_subspaces_with_range(node.children3, x1, y1, x2, y2, result_image)

    def overlaps(self, node, x1, y1, x2, y2):
        return not (x2 < node.x or x1 > node.x + node.width or y2 < node.y or y1 > node.y + node.height)

    def mask(self, x1, y1, x2, y2):
        masked_image = [row[:] for row in self.image]

        self._mask(self.root, x1, y1, x2, y2, masked_image)

        return masked_image

    def _mask(self, node, x1, y1, x2, y2, masked_image):
        if node is None or not self.overlaps(node, x1, y1, x2, y2):
            return

        if node.children0 == None and node.children1 == None and node.children2 == None and node.children3 == None:

            for i in range(node.x, node.x + node.width):
                for j in range(node.y, node.y + node.height):
                    masked_image[i][j] = (0, 0, 0)
            return

        self._mask(node.children0, x1, y1, x2, y2, masked_image)
        self._mask(node.children1, x1, y1, x2, y2, masked_image)
        self._mask(node.children2, x1, y1, x2, y2, masked_image)
        self._mask(node.children3, x1, y1, x2, y2, masked_image)

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

    def compress_depth(self, max_depth):
        #compressed_image = [[None for _ in range(len(self.image[0]))] for _ in range(len(self.image))]
        compressed_image = [[(0, 0, 0) for _ in range(len(self.image[0]))] for _ in range(len(self.image))]
        self._build_limited_depth(self.root, max_depth, 0, compressed_image)
        return compressed_image

    def _build_limited_depth(self, node, max_depth, current_depth, compressed_image):
        if node is None:
            return

        if current_depth == max_depth or (node.children0 == None and node.children1 == None and node.children2 == None and node.children3 == None):
            avg_color = self.average_color(node.x, node.y, node.width, node.height)

            for i in range(node.x, node.x + node.width):
                for j in range(node.y, node.y + node.height):
                    if 0 <= i < len(self.image) and 0 <= j < len(self.image[0]):
                        compressed_image[i][j] = avg_color
            return

        self._build_limited_depth(node.children0, max_depth, current_depth + 1, compressed_image)
        self._build_limited_depth(node.children1, max_depth, current_depth + 1, compressed_image)
        self._build_limited_depth(node.children2, max_depth, current_depth + 1, compressed_image)
        self._build_limited_depth(node.children3, max_depth, current_depth + 1, compressed_image)

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
        masked_frame1 = [list(map(list, row)) for row in masked_frame]
        masked_frame1 = np.array(masked_frame1, dtype=np.uint8)

        search_frame = quad_tree.search_subspaces_with_range(x1, y1, x2, y2)
        search_frame1 = [list(map(list, row)) for row in search_frame]
        search_frame1 = np.array(search_frame1, dtype=np.uint8)

        compress_frame = quad_tree.compress_depth(max_depth)
        compress_frame1 = [list(map(list, row)) for row in compress_frame]
        #compress_frame1 = [list(map(list, row)) for row in compress_frame if all(item is not None for item in row)]
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

def main():
    image = image_to_array(r"D:\personal\Photo\test.jpg")
    quad_tree = QuadTree(image)

    depth = quad_tree.tree_depth()
    print("Tree Depth:", depth)

    pixel_depth = quad_tree.pixel_depth(420, 450)
    print("Pixel Depth:", pixel_depth)

    new_image_array = quad_tree.search_subspaces_with_range(1, 1, 480, 480)
    new_image = quad_tree.array_to_image(new_image_array)
    new_image.save(r"D:\personal\Photo\new_image.jpg")

    masked_image_array = quad_tree.mask(1, 1, 480, 480)
    masked_image = quad_tree.array_to_image(masked_image_array)
    masked_image.save(r"D:\personal\Photo\masked_image.jpg")

    compressed_image_array = quad_tree.compress_depth(15)
    compressed_image = quad_tree.array_to_image(compressed_image_array)
    compressed_image.save(r"D:\personal\Photo\compressed_image.jpg")

    process_video(r"D:\personal\Photo\ftest.mp4", r"D:\personal\Photo\ftest1.avi", r"D:\personal\Photo\ftest2.avi", r"D:\personal\Photo\ftest3.avi", 2, 2, 480, 480, 9 )

if __name__ == "__main__":
    main()