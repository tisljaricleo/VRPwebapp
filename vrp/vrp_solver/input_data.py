from misc import save_pickle_data, distance_diff

data = [
    (0, 45.8242, 15.906),
    (1, 45.8135, 15.949),
    (2, 45.8129, 15.9594),
    (3, 45.8123, 15.9808),
    (4, 45.8161, 16.0105),
    (5, 45.819, 16.0505),
    (6, 45.7999, 15.9201),
    (7, 45.801, 15.9324),
    (8, 45.7982, 15.9642),
    (9, 45.7963, 15.9805),  # depot
    (10, 45.7964, 16.0152),
    (11, 45.7996, 16.0365),
    (12, 45.7988, 16.0787),
    (13, 45.781, 15.9071),
    (14, 45.7848, 15.9315),
    (15, 45.7824, 15.9783),
    (16, 45.7839, 16.0318),
    (17, 45.7847, 16.0636),
    (18, 45.7617, 15.906),
    (19, 45.7518, 15.9455),
    (20, 45.7519, 15.9929),
    (21, 45.7536, 16.0384)
]

points = list()

ex_counter = 0

for i in range(0, len(data)):

    points.append({'osm_id': None,
                   'id': i,
                   'lat': float(data[i][1]),
                   'lon': float(data[i][2])})


distance_matrix = list([])

for i in range(0, len(points)):

    curr = points[i]
    temp = list([])

    for j in range(0, len(points)):

        if i == j:
            temp.append(0)
            continue

        next_ = points[j]
        distance = distance_diff(curr['lat'], curr['lon'], next_['lat'], next_['lon'])

        temp.append(int(distance))

    distance_matrix.append(temp)

print()
save_pickle_data('small_distance_matrix.pkl', distance_matrix)
