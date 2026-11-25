"""Write a small .ico (indigo tile with a white focus ring) using only the
standard library, so the build needs no image library."""
import math
import struct
import sys


def _image(size):
    """32-bit BGRA bottom-up pixel rows + 1-bit AND mask, as an ICO BMP entry."""
    c = (size - 1) / 2
    radius_tile = size * 0.22
    rows = []
    for y in range(size - 1, -1, -1):          # BMP rows are bottom-up
        row = bytearray()
        for x in range(size):
            # rounded square tile
            dx, dy = max(abs(x - c) - (c - radius_tile), 0), max(abs(y - c) - (c - radius_tile), 0)
            inside = math.hypot(dx, dy) <= radius_tile
            d = math.hypot(x - c, y - c) / size
            if not inside:
                row += bytes((0, 0, 0, 0))
            elif 0.27 <= d <= 0.36 or d <= 0.1:    # ring + centre dot
                row += bytes((255, 255, 255, 255))
            else:
                row += bytes((0xA3, 0x30, 0x37, 255))   # #3730A3 as BGRA
        rows.append(bytes(row))
    pixels = b"".join(rows)
    mask_row = ((size + 31) // 32) * 4
    mask = bytes(mask_row * size)               # alpha channel decides transparency
    header = struct.pack("<IiiHHIIiiII", 40, size, size * 2, 1, 32, 0,
                         len(pixels) + len(mask), 0, 0, 0, 0)
    return header + pixels + mask


def write_ico(path, sizes=(16, 32, 48, 64)):
    images = [_image(s) for s in sizes]
    out = struct.pack("<HHH", 0, 1, len(images))
    offset = 6 + 16 * len(images)
    for s, img in zip(sizes, images):
        out += struct.pack("<BBBBHHII", s % 256, s % 256, 0, 0, 1, 32, len(img), offset)
        offset += len(img)
    with open(path, "wb") as f:
        f.write(out + b"".join(images))


if __name__ == "__main__":
    write_ico(sys.argv[1] if len(sys.argv) > 1 else "focus_tracker.ico")
