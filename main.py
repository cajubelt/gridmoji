import math
from typing import Optional, Any, Dict

from typing_extensions import Protocol, runtime_checkable

import emoji


class Region(Protocol):
    def __call__(self, row: int, col: int) -> bool:
        ...


class Grid(Protocol):
    def __call__(self, row: int, col: int) -> str:
        ...


def union(*regions: Region) -> Region:
    def output(row: int, col: int) -> bool:
        return any([region(row=row, col=col) for region in regions])

    return output


def intersection(*regions: Region) -> Region:
    def output(row: int, col: int) -> bool:
        return all([region(row=row, col=col) for region in regions])

    return output


def horiz_line(row_idx: int) -> Region:
    return line(rows_per_col=0, start_row=row_idx, start_col=0)


def vert_line(idx: int) -> Region:
    def output(row: int, col: int) -> bool:
        return col == idx

    return output


def line(rows_per_col: float, start_row: int, start_col: int) -> Region:
    def output(row: int, col: int) -> bool:
        diff = (row - start_row) - (rows_per_col * (col - start_col))
        return math.floor(diff) == 0
    return output


def circle(center_row: int, center_col: int, radius: int) -> Region:
    def output(row: int, col: int) -> bool:
        return math.fabs(center_row - row) + math.fabs(center_col - col) == radius

    return output


def box_boundary(row_start: int, height: int, col_start: int, width: int) -> Region:
    def output(row: int, col: int) -> bool:
        return (
                row_start <= row < row_start + height and
                col_start <= col < col_start + width
        )

    return output


class BoxShape(Protocol):
    def __call__(self, row_start: int, height: int, col_start: int, width: int) -> Region:
        ...


class RegionCallback(Protocol):
    def __call__(self, *region: Region) -> Region:
        ...


def combine_box_shapes(region_callback: RegionCallback,
                       *box_shapes: BoxShape) -> BoxShape:
    def output(
            row_start: int, height: int, col_start: int, width: int
    ) -> Region:
        return region_callback(*[box_shape(row_start=row_start,
                                           width=width,
                                           col_start=col_start,
                                           height=height)
                                 for box_shape in box_shapes])

    return output


def box_shape_union(*box_shapes: BoxShape) -> BoxShape:
    return combine_box_shapes(union, *box_shapes)


def box_shape_intersection(*box_shapes: BoxShape) -> BoxShape:
    return combine_box_shapes(intersection, *box_shapes)


def inverse_region(region: Region) -> Region:
    def not_in_region(row: int, col:int) -> bool:
        return not region(row=row, col=col)
    return not_in_region


def inverse_box(box_shape: BoxShape) -> BoxShape:
    def output_box(row_start: int, height: int, col_start: int, width: int) -> Region:
        return inverse_region(box_shape(row_start=row_start,
                                        height=height,
                                        col_start=col_start,
                                        width=width))
    return output_box


def in_box(box_shape: BoxShape) -> BoxShape:
    return box_shape_intersection(box_boundary, box_shape)


@in_box
def box_seg_overline(row_start: int, height: int, col_start: int, width: int) -> Region:
    return horiz_line(row_start)


@in_box
def box_seg_underline(row_start: int, height: int, col_start: int, width: int) -> Region:
    return horiz_line(row_start + height - 1)


@in_box
def box_seg_left_vert(row_start: int, height: int, col_start: int, width: int) -> Region:
    return vert_line(col_start)


@in_box
def box_seg_right_vert(row_start: int, height: int, col_start: int, width: int) -> Region:
    return vert_line(col_start + width - 1)


@in_box
def box_seg_mid_horiz(row_start: int, height: int, col_start: int, width: int) -> Region:
    return horiz_line(row_start + height // 2)


@in_box
def box_seg_forward_slash(row_start: int, height: int, col_start: int, width: int) -> Region:
    return line(
        rows_per_col=-1 * height / width,
        start_row=row_start + height - 1,
        start_col=col_start
    )


@in_box
def box_seg_lower_back_slash(row_start: int, height: int, col_start: int, width: int) -> Region:
    rows_down = height // 2
    return line(
        start_row=row_start + rows_down,
        rows_per_col=(height - rows_down) / width,
        start_col=col_start
    )


@in_box
def box_seg_right_semicircle(row_start: int, height: int, col_start: int, width: int) -> Region:
    return circle(
        center_row=row_start + height // 2,
        center_col=col_start,
        radius=min(height // 2, width - 1)
    )


@in_box
def upper_half_boundary(row_start: int, height: int, col_start: int, width: int) -> Region:
    def upper_half_region(row: int, col: int) -> bool:
        return row <= row_start + height // 2
    return upper_half_region


lower_half_boundary = box_shape_intersection(box_boundary, inverse_box(upper_half_boundary))


def in_upper_half(box_shape: BoxShape) -> BoxShape:
    return box_shape_intersection(box_shape, upper_half_boundary)


def in_lower_half(box_shape: BoxShape) -> BoxShape:
    return box_shape_intersection(box_shape, lower_half_boundary)


@in_upper_half
def box_seg_upper_left_backslash(row_start: int, height: int, col_start: int, width: int) -> Region:
    return line(rows_per_col=math.ceil(height / width / 2),
                start_row=row_start,
                start_col=col_start)  # fixme use reflection for this instead


box_seg_upper_right_vert = in_upper_half(box_seg_right_vert)


box_seg_upper_left_vert = in_upper_half(box_seg_left_vert)


box_seg_lower_right_vert = in_lower_half(box_seg_right_vert)


@in_lower_half
def box_seg_lower_left_forward_slash(row_start: int, height: int, col_start: int, width: int) -> Region:
    return line(
        start_row=row_start + height - 1,
        start_col=col_start,
        rows_per_col=-1 * math.floor(height / width)
    )


@in_box
def box_seg_mid_vert(row_start: int, height: int, col_start: int, width: int) -> Region:
    return vert_line(col_start + width // 2)


def reflect_top_bottom(box_shape: BoxShape) -> BoxShape:
    """
    Get a box shape "reflected vertically" over a horizontal midline. (Top-bottom reflection)
    """
    def reflected_box(row_start: int, height: int, col_start: int, width: int) -> Region:
        region = box_shape(row_start=row_start, height=height, col_start=col_start, width=width)

        def reflected_region(row: int, col: int) -> bool:
            return region(row=(row_start + height - 1) - (row - row_start),
                          col=col)
        return reflected_region
    return reflected_box


def reflect_left_right(box_shape: BoxShape) -> BoxShape:
    """
    Get a box shape "reflected horizontally" over a vertical midline. (Left-right reflection)
    """
    def reflected_box(row_start: int, height: int, col_start: int, width: int) -> Region:
        region = box_shape(row_start=row_start, height=height, col_start=col_start, width=width)

        def reflected_region(row: int, col: int) -> bool:
            return region(row=row,
                          col=(col_start + width - 1) - (col - col_start))
        return reflected_region
    return reflected_box


box_seg_lower_left_vert = reflect_top_bottom(box_seg_upper_left_vert)


box_seg_back_slash = reflect_left_right(box_seg_forward_slash)


box_seg_upper_forward_slash = reflect_top_bottom(box_seg_lower_back_slash)


box_seg_upper_right_forward_slash = reflect_left_right(box_seg_upper_left_backslash)


def top_right_point(row_start: int, height: int, col_start: int, width: int) -> Region:
    def is_top_right(row: int, col: int) -> bool:
        return (row == row_start) and (col == (col_start + width - 1))
    return is_top_right


def without_top_right(box_shape: BoxShape) -> BoxShape:
    return box_shape_intersection(box_shape, inverse_box(top_right_point))


def without_bottom_right(box_shape: BoxShape) -> BoxShape:
    return reflect_top_bottom(without_top_right(reflect_top_bottom(box_shape)))


def empty_region(row: int, col: int) -> bool:
    return False


def empty_box_shape(row_start: int, height: int, col_start: int, width: int) -> Region:
    return empty_region


letter_m = box_shape_union(box_seg_left_vert,
                           box_seg_right_vert,
                           box_seg_upper_right_forward_slash,
                           box_seg_upper_left_backslash)

letter_o = box_shape_union(box_seg_left_vert, box_seg_underline, box_seg_overline, box_seg_right_vert)
ALPHABET: Dict[str, BoxShape] = {
    'A': box_shape_union(box_seg_overline,
                         box_seg_left_vert,
                         box_seg_right_vert,
                         box_seg_mid_horiz),
    'B': box_shape_union(box_seg_left_vert,
                         box_seg_overline,
                         box_seg_underline,
                         box_seg_mid_horiz,
                         box_seg_right_vert),
    'C': box_shape_union(box_seg_left_vert,
                         box_seg_overline,
                         box_seg_underline),
    'D': without_bottom_right(without_top_right(letter_o)),
    'E': box_shape_union(box_seg_left_vert,
                         box_seg_mid_horiz,
                         box_seg_underline,
                         box_seg_overline),
    'F': box_shape_union(box_seg_left_vert,
                         box_seg_overline,
                         box_seg_mid_horiz),
    'G': box_shape_union(box_seg_right_vert,
                         box_seg_overline,
                         box_seg_underline,
                         box_seg_mid_horiz,
                         box_seg_underline,
                         box_seg_upper_left_vert),
    'H': box_shape_union(box_seg_left_vert,
                         box_seg_right_vert,
                         box_seg_mid_horiz),
    'I': box_shape_union(box_seg_mid_vert,
                         box_seg_overline,
                         box_seg_underline),
    'J': box_shape_union(box_seg_right_vert,
                         box_seg_underline,
                         box_seg_lower_left_vert),
    'K': box_shape_union(box_seg_left_vert,
                         box_seg_lower_back_slash,
                         box_seg_upper_forward_slash),
    'L': box_shape_union(box_seg_left_vert,
                         box_seg_underline),
    'M': letter_m,
    'N': box_shape_union(box_seg_left_vert,
                         box_seg_right_vert,
                         box_seg_overline),
    'O': letter_o,
    'P': box_shape_union(box_seg_left_vert,
                         box_seg_overline,
                         box_seg_mid_horiz,
                         box_seg_upper_right_vert),
    'Q': box_shape_union(box_seg_upper_left_vert,
                         box_seg_right_vert,
                         box_seg_overline,
                         box_seg_mid_horiz),
    'R': box_shape_union(box_seg_overline,
                         box_seg_mid_horiz,
                         box_seg_upper_right_vert,
                         box_seg_left_vert,
                         box_seg_lower_back_slash),
    'S': box_shape_union(box_seg_overline,
                         box_seg_underline,
                         box_seg_mid_horiz,
                         box_seg_upper_left_vert,
                         box_seg_lower_right_vert
                         ),
    'T': box_shape_union(box_seg_mid_vert,
                         box_seg_overline),
    'U': box_shape_union(box_seg_left_vert,
                         box_seg_underline,
                         box_seg_right_vert),
    'V': box_shape_union(box_seg_right_vert,
                         box_seg_back_slash),
    'W': reflect_top_bottom(letter_m),
    'X': box_shape_union(box_seg_forward_slash,
                         box_seg_back_slash),
    'Y': box_shape_union(box_seg_left_vert,
                         box_seg_upper_forward_slash),
    'Z': box_shape_union(box_seg_overline,
                         box_seg_underline,
                         box_seg_forward_slash),
    ' ': empty_box_shape
}

DEFAULT_LETTER_WIDTH = 5
DEFAULT_LETTER_HEIGHT = 5
DEFAULT_HORIZ_SPACING = 1
DEFAULT_VERT_SPACING = 1


def get_grid(word: str,
             bg_emoji_alias: str,
             text_emoji_alias: str,
             letter_width: int,
             letter_height: int) -> Grid:
    # todo could use an enum for aliases to throw smarter errors
    chars = [char for char in word.upper()]
    box_shapes = [ALPHABET[char] for char in chars]
    col_starts = [idx * (DEFAULT_LETTER_WIDTH + DEFAULT_HORIZ_SPACING) + DEFAULT_HORIZ_SPACING
                  for idx in range(len(word))]
    regions = [box_shape(row_start=DEFAULT_VERT_SPACING,
                         height=letter_height,
                         col_start=col_start,
                         width=letter_width) for box_shape, col_start in zip(box_shapes, col_starts)]
    grid_region = union(*regions)

    def output(row: int, col: int) -> str:
        return text_emoji_alias if grid_region(row, col) else bg_emoji_alias

    return output


def print_grid(word: str,
               bg_emoji_alias: str = ':blue_circle:',
               text_emoji_alias: str = ':red_circle:',
               letter_width: int = DEFAULT_LETTER_WIDTH,
               letter_height: int = DEFAULT_LETTER_HEIGHT
               ) -> None:
    grid = get_grid(word,
                    bg_emoji_alias=bg_emoji_alias,
                    text_emoji_alias=text_emoji_alias,
                    letter_width=letter_width,
                    letter_height=letter_height)
    grid_height = letter_height + 2 * DEFAULT_VERT_SPACING
    grid_width = (letter_width + DEFAULT_HORIZ_SPACING) * len(word) + DEFAULT_HORIZ_SPACING
    for row_idx in range(grid_height):
        row = ''
        for col_idx in range(grid_width):
            row += (grid(row_idx, col_idx))
        print(emoji.emojize(row, use_aliases=True))


'''
GRID is a function taking a coordinate pair and returning a string that should appear at that location. 

REGION is a function taking coordinate pair and returning a boolean indicating whether the point falls in a 
    constrained area
    
BOX_SHAPE is a function taking a coordinate pair for the upper-left corner of the box_boundary (counted from upper-left corner
    of the grid), width, and height, and returning a REGION representing all points within the box_boundary
'''

if __name__ == '__main__':
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    print_grid('dude', text_emoji_alias=':green_circle:')
