import json
import math
from misc import save_pickle_data, distance_diff


# def get_distance_matrix():


with open('export.json', encoding="utf8") as f:
    data = json.load(f)

points = list()

ex_counter = 0

for i in range(0, len(data['elements'])):
    try:
        points.append({'osm_id': data['elements'][i]['id'],
                       'id': i,
                       'lat': data['elements'][i]['lat'],
                       'lon': data['elements'][i]['lon']})
    except:
        ex_counter += 1


distance_matrix = list([])

for i in range(0, len(points)):

    curr = points[i]
    temp = list([])

    for j in range(0, len(points)):

        if i == j:
            temp.append(0)
            continue

        next_ = points[j]
        distance = distance_diff(curr['lat'], curr['lon'], next_[
                                 'lat'], next_['lon'])
        temp.append(distance)

    distance_matrix.append(temp)


save_pickle_data('distance_matrix.pkl',distance_matrix)
