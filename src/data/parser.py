import json


class Parser:
    def __init__(self, json_path):
        self.json_path = json_path
    
    def parse(self):
        with open(self.json_path) as f:
            data = json.load(f)

        parsed_cells = []
        for img_data in data.values():
            for region in img_data['regions']:
                label= region['region_attributes']['cell']
                shape = region['shape_attributes']
                if(shape['name'] == 'polygon'):
                    points_x = shape['all_points_x']
                    points_y = shape['all_points_y']

                    parsed_cells.append({
                        'filename': img_data['filename'],
                        'label': label,
                        'points_x': points_x,
                        'points_y': points_y
                    })

        return parsed_cells