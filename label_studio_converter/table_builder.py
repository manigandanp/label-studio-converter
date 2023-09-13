import numpy as np
from functools import reduce

def overlaps(bbox1, bbox2, threshold=0.5):
    """
    Test if more than "threshold" fraction of bbox1 overlaps with bbox2.
    """
    rect1 = Rect(bbox1)
    area1 = rect1.area()
    if area1 == 0:
        return False
    intersection = rect1.intersect(Rect(bbox2))
    if intersection is None:
        return False, None
    return intersection.area() / area1 >= threshold, intersection.area() / area1

def convert_percentage_to_bbox(percent_bbox, page_width, page_height):
    left_percentage = percent_bbox["x"]
    top_percentage = percent_bbox["y"]
    width_percentage = percent_bbox["width"]
    height_percentage = percent_bbox["height"]

    x1 = left_percentage * page_width / 100.0
    y1 = page_height - (top_percentage * page_height / 100.0)
    x2 = x1 + (width_percentage * page_width / 100.0)
    y2 = y1 - (height_percentage * page_height / 100.0)

    return camelot_to_pascal_bbox((x1, y1, x2, y2), page_height)
  
def camelot_to_pascal_bbox(bbox, page_height):
    xmin, ymin, xmax, ymax = bbox
    x0 = xmin
    y0 = page_height - ymin
    x1 = xmax
    y1 = page_height - ymax
    return [x0, y0, x1, y1]

def prepare_table_data(task):
    updated_results = task.get('annotations') or task.get('drafts')
    if not updated_results:
        return None

    def filter_and_convert(label):
        return [
            convert_percentage_to_bbox(r['value'], r['original_width'], r['original_height'])
            for r in updated_results[0]['result'] 
            if label in r['value']['rectanglelabels']
        ]

    rows = filter_and_convert('table row')
    columns = filter_and_convert('table column')
    headers = filter_and_convert('table column header')
    tables = filter_and_convert('table')
    new_rows = headers + rows
    return new_rows, columns, headers, tables
  
def post_process_table(header, rows, columns, table_bbox):
  structure = {
        'columns': [{'normalized_scaled_bbox': c} for c in columns],
        'rows': [{'normalized_scaled_bbox' : row['row_bbox']} for row in rows], #table['structure']['rows']
        'headers' : [{'normalized_scaled_bbox' : header['row_bbox']}] if header is not None else []
    }
    
  result_table = dict({'table_bbox': table_bbox,
                        'header': header,
                        'rows': [[cell['cell'] for cell in row['row']]for row in rows], 
                        'row_bboxes': [row['row_bbox'] for row in rows],
                        'structure' : structure
                        })
  return result_table


class TableBuilder:
    def __init__(self):
        self.table = None

    def build_table(self, rows, columns, words, table_bbox, headers):
        def check_overlap(r):
          is_overlapping, _= overlaps(r, table_bbox)
          return is_overlapping
        
        rows = [r for r in rows if check_overlap(r)]
        columns = [r for r in columns if check_overlap(r)]
        headers = [r for r in headers if check_overlap(r)]
 
        cells = self.get_cell_bboxes(rows, columns)
        words_within_table = self.filter_words_within_bbox(table_bbox, words)
        
        cell_words = self.assign_words_to_cells(
            words_within_table, rows, cells)
        result_table = self.build_table_from_cells(cell_words, cells)
        if len(result_table) > 0:
            result_table,  empty_rows_indices, empty_cols_indices = self.remove_empty_rows_and_columns(
                result_table)
            columns = [bbox for i, bbox in enumerate(
                columns) if i not in empty_cols_indices]  # removing empty column bboxes
            header, rows = self.split_header_and_rows(headers, result_table)
            
            return header, rows, columns
        else:
            return None, [], []

    def filter_words_within_bbox(self, bbox, words):
        return [word for word in words if overlaps([word['x0'], word['top'], word['x1'], word['bottom']], bbox, 0.1)]

    def get_cell_bboxes(self, rows, columns):
        # Sort rows and columns
        rows = sorted(rows, key=lambda x: x[1], reverse=False)
        columns = sorted(columns, key=lambda x: x[0])

        cell_bboxes = []
        for row in rows:
            row_bboxes = []
            for col in columns:
                # Get the intersection of the row and column
                xmin = max(row[0], col[0])
                ymin = max(row[1], col[1])
                xmax = min(row[2], col[2])
                ymax = min(row[3], col[3])
                row_bboxes.append([xmin, ymin, xmax, ymax])
            cell_bboxes.append(row_bboxes)

        return cell_bboxes

    def assign_word_to_table_row_cell(self, word_bbox, rows, cells):
        """
        Assigns a word to the cell with which it has the maximum overlap.

        Args:
        word (list): The bbox of the word.
        rows (list): A list containing the bboxes of the rows.
        cells (list): A 2D list containing the bboxes of the cells.

        Returns:
        max_overlap_cell (list): The bbox of the cell with the maximum overlap.
        """
        max_overlap = 0
        max_overlap_cell = None

        for row, cells in zip(rows, cells):
            row_overlap, _ = overlaps(word_bbox, row)
            if row_overlap:
                # Finally, within that row, find the cell that the word belongs to
                for cell in cells:
                    cell_overlap, fraction = overlaps(word_bbox, cell)
                    if cell_overlap and fraction > max_overlap:
                        max_overlap = fraction
                        max_overlap_cell = cell

        return max_overlap_cell

    def assign_words_to_cells(self, words, rows, cells):
        """
        Assigns each word to the cell with which it has the maximum overlap.

        Args:
        words (list): A list of words, where each word is a dictionary with keys 'x0', 'top', 'x1', 'bottom', and 'text'.
        rows (list): A list containing the bboxes of the rows.
        cells (list): A 2D list containing the bboxes of the cells.

        Returns:
        cell_words (dict): A dictionary mapping cells to lists of words.
        """
        # Create a dictionary to hold words for each cell
        cell_words = {tuple(cell): [] for row in cells for cell in row}

        for word in words:
            word_bbox = [word['x0'], word['top'], word['x1'], word['bottom']]
            # Assign the word to the cell with the maximum overlap
            cell = self.assign_word_to_table_row_cell(word_bbox, rows, cells)
            if cell is not None:
                cell_words[tuple(cell)].append((word['text'], word_bbox))
        return cell_words

    def build_table_from_cells(self, cell_words, cells, y_tolerance=5):
        """
        Builds a table from the given cells and words.

        Args:
        cell_words (dict): A dictionary mapping cells to lists of words.
        cells (list): A 2D list containing the bboxes of the cells.
        y_tolerance (int): Tolerance in y-coordinate to consider words are on the same line.

        Returns:
        result_table (list): A 2D list representing the table, where each element is a dictionary representing a row.
        """
        def group_words_by_y_coordinate(row, cell_words, y_tolerance):
            row_lines = []
            for cell in row:
                words = cell_words[tuple(cell)]
                lines = group_words_in_same_line(words, y_tolerance)
                sorted_lines = sort_lines_by_y_coordinate(lines)
                row_lines.append(sorted_lines)
            return row_lines

        def group_words_in_same_line(words, y_tolerance):
            lines = {}
            for word, bbox in words:
                line_key = next((k for k, v in lines.items() if abs(
                    bbox[1]-v[0][1][1]) < y_tolerance), None)
                if line_key is None:
                    lines[bbox[1]] = [(word, bbox)]
                else:
                    lines[line_key].append((word, bbox))
            return lines

        def sort_lines_by_y_coordinate(lines):
            return sorted([sorted(line, key=lambda x: x[1][0]) for line in lines.values()], key=lambda x: x[0][1][1])

        def add_new_rows_for_each_line(result_table, row, max_lines):
            for _ in range(max_lines):
                # Initialize row_bbox with first cell's bbox
                result_table.append({'row': [], 'row_bbox': list(row[0])})
            return result_table

        def add_cells_to_table(result_table, row_lines, max_lines, row):
            for i, sorted_lines in enumerate(row_lines):
                for j, line in enumerate(sorted_lines):
                    cell_bbox = get_bbox_of_line(line)
                    sentence = " ".join([w[0] for w in line])
                    word_bboxes = [w[1] for w in line]
                    result_table[-max_lines+j]['row'].append(
                        {'cell': sentence, 'cell_bbox': cell_bbox, 'word_bboxes': word_bboxes})
                for j in range(len(sorted_lines), max_lines):
                    # Initialize with current cell's bbox
                    cell_bbox = row[i].copy()
                    result_table[-max_lines+j]['row'].append(
                        {'cell': '', 'cell_bbox': cell_bbox, 'word_bboxes': []})
                result_table = update_bbox_of_row(result_table, max_lines, i)
            return result_table

        def get_bbox_of_line(line):
            return [line[0][1][0], line[0][1][1], line[-1][1][2], line[-1][1][3]]

        def update_bbox_of_row(result_table, max_lines, i):
            for j in range(max_lines):
                row = result_table[-max_lines+j]['row']
                min_x = min(cell['cell_bbox'][0] for cell in row)
                min_y = min(cell['cell_bbox'][1] for cell in row)
                max_x = max(cell['cell_bbox'][2] for cell in row)
                max_y = max(cell['cell_bbox'][3] for cell in row)
                result_table[-max_lines +
                             j]['row_bbox'] = [min_x, min_y, max_x, max_y]
            return result_table

        result_table = []

        for row in cells:
            row_lines = group_words_by_y_coordinate(
                row, cell_words, y_tolerance)
            max_lines = max(len(lines) for lines in row_lines)
            result_table = add_new_rows_for_each_line(
                result_table, row, max_lines)
            result_table = add_cells_to_table(
                result_table, row_lines, max_lines, row)
        return result_table

    def get_row_bbox_from_cells(self, bboxes):
        min_x = min(bbox[0] for bbox in bboxes)
        max_x = max(bbox[2] for bbox in bboxes)

        # Assuming y coordinates are consistent across all bboxes
        min_y = bboxes[0][1]
        max_y = bboxes[0][3]

        return [min_x, min_y, max_x, max_y]

    def remove_empty_rows_and_columns(self, data):
        # Convert the data to a 2D NumPy array
        data_array = np.array([[cell['cell']
                              for cell in row['row']] for row in data])
        # Create a boolean mask for non-empty rows and columns
        non_empty_rows_mask = np.array(
            [len(''.join(row)) > 0 for row in data_array])
        non_empty_cols_mask = np.array(
            [len(''.join(col)) > 0 for col in data_array.T])
        # Identify empty rows and columns
        empty_rows_indices = np.where(~non_empty_rows_mask)[0].tolist()
        empty_cols_indices = np.where(~non_empty_cols_mask)[0].tolist()

        # Filter the original data using the boolean masks
        filtered_data = [row for i, row in enumerate(
            data) if non_empty_rows_mask[i]]
        for row in filtered_data:
            row['row'] = [cell for i, cell in enumerate(
                row['row']) if non_empty_cols_mask[i]]

        return filtered_data, empty_rows_indices, empty_cols_indices

    def split_header_and_rows(self, headers, rows):
        def reducer(buckets, row):
            row_bbox = row['row_bbox']
            has_overlap, _ = overlaps(row_bbox, header_bbox)
            if has_overlap:
                buckets[0].append(row)  # header rows
            else:
                buckets[1].append(row)  # other rows
            return buckets

        if len(headers) > 0:
            header_bbox = headers[0]
            header, rows = reduce(reducer, rows, ([], []))
            header = {'row': self.concat_header_rows(
                header, sep=" "), 'row_bbox': header_bbox}
            return header, rows
        return (None, rows)

    def concat_header_rows(self, headers_ip, sep="\n"):
        # TODO: Add cell_bboxex to header
        headers = [[cell['cell'] for cell in row['row']] for row in headers_ip]
        return [sep.join(r).strip() for r in zip(*headers)]
  
class Rect:
    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], (list, tuple)) and len(args[0]) == 4:
            left, top, right, bottom = args[0]
        elif len(args) == 4:
            left, top, right, bottom = args
        else:
            raise ValueError("Invalid arguments")
        self.left = float(left)
        self.top = float(top)
        self.right = float(right)
        self.bottom = float(bottom)

    def __repr__(self):
        return f"Rect({self.left}, {self.top}, {self.right}, {self.bottom})"

    def width(self):
        return self.right - self.left

    def height(self):
        return self.bottom - self.top

    def area(self):
        return self.width() * self.height()

    def contains(self, other):
        return (
            self.left <= other.left
            and self.top <= other.top
            and self.right >= other.right
            and self.bottom >= other.bottom
        )

    def contains_point(self, point):
        x, y = point
        return (
            self.left <= x <= self.right
            and self.top <= y <= self.bottom
        )

    def intersect(self, other):
        left = max(self.left, other.left)
        top = max(self.top, other.top)
        right = min(self.right, other.right)
        bottom = min(self.bottom, other.bottom)
        if right < left or bottom < top:
            return None
        return Rect(left, top, right, bottom)

    def union(self, other):
        left = min(self.left, other.left)
        top = min(self.top, other.top)
        right = max(self.right, other.right)
        bottom = max(self.bottom, other.bottom)
        return Rect(left, top, right, bottom)

    def normalize(self):
        left, top, right, bottom = self.left, self.top, self.right, self.bottom
        if left > right:
            left, right = right, left
        if top > bottom:
            top, bottom = bottom, top
        return Rect(left, top, right, bottom)

    def round(self):
        return Rect(
            round(self.left),
            round(self.top),
            round(self.right),
            round(self.bottom),
        )

