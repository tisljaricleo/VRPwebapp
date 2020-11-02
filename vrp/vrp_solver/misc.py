import datetime
import pickle
# import config
import numpy as np
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageDraw
from math import sqrt, pow
import math


def distance_diff(lat1, lon1, lat2, lon2):
    '''
        Implementation of Haversine formula for calculating distance between two points on a sphere
        https://en.wikipedia.org/wiki/Haversine_formula
    '''
    R = 6378.137  # Radius of earth in KM
    dLat = lat2 * math.pi / 180 - lat1 * math.pi / 180
    dLon = lon2 * math.pi / 180 - lon1 * math.pi / 180
    a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(lat1 * math.pi / 180) * \
        math.cos(lat2 * math.pi / 180) * math.sin(dLon/2) * math.sin(dLon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = R * c
    return d * 1000  # meters

def get_anomaly_type(matrix, sl=70):
    res = config.RESOLUTION
    max_index = config.MAX_INDEX - 1
    fe = (0, int(0.4*sl/res))
    d = (fe[1]+1, int(0.5*sl/res))
    c = (d[1] + 1, int(0.67 * sl / res))
    b = (c[1] + 1, int(0.85 * sl / res))
    a = (b[1] + 1, max_index)

    ccc = matrix[fe[0]:fe[1], fe[0]:fe[1]]

    sums = dict({'fe_fe': np.sum(matrix[fe[0]:fe[1], fe[0]:fe[1]]),
                 'fe_d': np.sum(matrix[fe[0]:fe[1], d[0]:d[1]]),
                 'fe_c': np.sum(matrix[fe[0]:fe[1], c[0]:c[1]]),
                 'fe_b': np.sum(matrix[fe[0]:fe[1], b[0]:b[1]]),
                 'fe_a': np.sum(matrix[fe[0]:fe[1], a[0]:a[1]]),
                 'd_fe': np.sum(matrix[d[0]:d[1], fe[0]:fe[1]]),
                 'c_fe': np.sum(matrix[c[0]:c[1], fe[0]:fe[1]]),
                 'b_fe': np.sum(matrix[b[0]:b[1], fe[0]:fe[1]]),
                 'a_fe': np.sum(matrix[a[0]:a[1], fe[0]:fe[1]])})

    m = 0
    t = ''
    for key, value in sums.items():
        if value > m:
            m = value
            t = key

    return t


def get_anomaly(matrix, sl=70):
    res = config.RESOLUTION
    max_index = config.MAX_INDEX - 1

    FE = (0, int(0.4*sl/res))
    DC = (FE[1]+1, int(0.67 * sl / res))
    BA = (DC[1] + 1, max_index)

    three = np.sum(matrix[FE[0]:FE[1], :]) + np.sum(matrix[:, FE[0]:FE[1]]) - np.sum(matrix[FE[0]:FE[1], FE[0]:FE[1]])
    two = np.sum(matrix[DC[0]:DC[1], :]) + np.sum(matrix[:, DC[0]:DC[1]]) - np.sum(matrix[DC[0]:DC[1], DC[0]:DC[1]])
    one = np.sum(matrix[BA[0]:BA[1], :]) + np.sum(matrix[:, BA[0]:BA[1]]) - np.sum(matrix[BA[0]:BA[1], BA[0]:BA[1]])

    return round(one, 5), round(two, 5), round(three, 5)


def scale_nums(x, min, max, a, b):
    return (((b - a) * (x - min)) / (max - min)) + a


def get_mass_center(matrix):

    im = Image.fromarray(np.uint8(matrix))


    im = im.point(lambda x: 0 if x < 30 else x)
    plt.imshow(im)

    # Oznake na x i y osi se nazivaju ticks.
    loc = range(0, 20, 1)   # Prvo treba definirati lokacije za ticks. Pošto je naša matrica 20x20 lokacije idu 0-19 (isto je za obje osi).
    ticks_x = [str((i+1)*5) for i in loc]  # Ono što ce biti napisano. U nasem slucaju 5, 10, 15, ... 100
    plt.xticks(loc, ticks_x, rotation='vertical')  # Postavljanje ticks na x os.
    plt.yticks(loc, ticks_x)  # Postavljanje ticks na y os.
    plt.xlabel('Destination speed (km/h)')
    plt.ylabel('Origin speed (km/h)')

    plt.show()

    # im = Image.open(image_path)
    immat = im.load()
    (X, Y) = im.size
    m = np.zeros((X, Y))

    for x in range(X):
        for y in range(Y):
            m[x, y] = immat[(x, y)] != (0)




    m = m / np.sum(np.sum(m))

    # marginal distributions
    dx = np.sum(m, 1)
    dy = np.sum(m, 0)

    # expected values
    cx = np.sum(dx * np.arange(X))
    cy = np.sum(dy * np.arange(Y))


    # draw = ImageDraw.Draw(im)
    # draw.point((cx, cy), 'red')

    # im.save('testfig.')
    # im.show()

    return cx, cy, np.array(im)


def kld(a, b):
    a[a <= 0] = 0.000000000001
    b[b <= 0] = 0.000000000001
    a = a.astype('float')
    b = b.astype('float')
    return np.sum(np.where(a != 0, a * np.log(a / b), 0))


# def get_anomaly(matrix):
#     # matrix = np.where(matrix < 20, 0, matrix)
#     # matrix = np.where(matrix > 0, 1, matrix)
#     # max_val_index = np.where(matrix == np.max(matrix))
#     # row = int(max_val_index[0][0])
#     # column = int(max_val_index[1][0])
#     # n_half = int(matrix.shape[0]/2)  # nxn matrix
#
#     n_EF = 0
#     # TODO: Count values over speed limit
#     n_over_sl = 0
#     # TODO: Count most likely transition (Los F to Los A)
#     # n_normal = 0
#     # n_q3 = 0
#     # n_q4 = 0
#
#     # TODO: bolje napisati s numpy slice npr. a[:, 3] i provjeriti je li brze
#     for row in range(0, matrix.shape[0]):
#         for column in range(0, matrix.shape[1]):
#             if matrix[row, column] > 0:
#                 if row < 4 or column < 4:
#                     n_EF += matrix[row, column]
#                 # else:
#                 #     n_normal += matrix[row, column]
#
#     # quadrants = [n_q1, n_q2, n_q3, n_q4]
#     # return quadrants.index(max(quadrants)) + 1
#     all_sum = np.sum(matrix)
#     anomaly_prob = float(n_EF) / float(all_sum)
#
#     return round(anomaly_prob * 100, 5)


def get_paths(folder_path, extension):
    paths = list([])
    for file in os.listdir(folder_path):
        if file.endswith(extension):
            paths.append(os.path.join(folder_path, file))
    return paths


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_time():
    return datetime.datetime.now()


def utc_to_local(utc_time):
    """
    Converts UTC time stamp to local time.
    :param utc_time: UTC timestamp.
    :return: Local time.
    """
    local_time = datetime.datetime.fromtimestamp(utc_time)
    return local_time


def get_date_parts(time_utc):
    """
    Returns year, month, week, day, summer, working_day from utc.
    'week': Week number of the year,
    'working_day': 0 - weekend, 1 - working day,
    'day': Monday = 0, Sunday = 6,
    'month': 1 - 12,
    'year':
    :param time_utc: Time in UTC format.
    :return:
    """
    # TODO: Check https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior

    local_time = datetime.datetime.fromtimestamp(time_utc)
    month = local_time.month
    year = local_time.year
    week = int(local_time.strftime("%W"))
    day = local_time.weekday()

    if month < 7 or month > 8:
        summer = 0
        if day < 5:
            working_day = 1
        else:
            working_day = 0
    else:
        summer = 1
        if day < 5:
            working_day = 1
        else:
            working_day = 0

    return year, month, week, day, summer, working_day


def interval_sep(time_utc):
    """
    Split to intervals
    :param time_utc: utc time
    :return: interval index 1 - 7
    """
    i1_lower = datetime.time(hour=5, minute=30)
    i1_higher = datetime.time(hour=6, minute=45)
    i2_lower = datetime.time(hour=6, minute=46)
    i2_higher = datetime.time(hour=7, minute=25)
    i3_lower = datetime.time(hour=7, minute=26)
    i3_higher = datetime.time(hour=8, minute=20)
    i4_lower = datetime.time(hour=8, minute=21)
    i4_higher = datetime.time(hour=15, minute=30)
    i5_lower = datetime.time(hour=15, minute=31)
    i5_higher = datetime.time(hour=17, minute=5)
    i6_lower = datetime.time(hour=17, minute=6)
    i6_higher = datetime.time(hour=19, minute=0)
    i7_lower = datetime.time(hour=19, minute=1)
    i7_higher = datetime.time(hour=22, minute=0)

    time = utc_to_local(time_utc).time()
    interval_index = 0

    if i1_lower < time < i1_higher:
        interval_index = 1
    if i2_lower < time < i2_higher:
        interval_index = 2
    if i3_lower < time < i3_higher:
        interval_index = 3
    if i4_lower < time < i4_higher:
        interval_index = 4
    if i5_lower < time < i5_higher:
        interval_index = 5
    if i6_lower < time < i6_higher:
        interval_index = 6
    if i7_lower < time < i7_higher:
        interval_index = 7

    return interval_index


def p_e_dist(x1, y1, x2, y2):
    return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))


def save_pickle_data(path, data):
    """
    Saves data in the pickle format.
    :param path: Path to save.
    :param data: Data to save.
    :return:
    """
    try:
        with open(path, 'wb') as handler:
            pickle.dump(data, handler, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        if hasattr(e, 'message'):
            print(e.message)
        else:
            print(e)


def open_pickle(path):
    """
    Opens pickle data from defined path.
    :param path: Path to pickle file.
    :return:
    """
    try:
        with open(path, 'rb') as handle:
            data = pickle.load(handle)
            return data
    except Exception as e:
        if hasattr(e, 'message'):
            print(e.message)
            return None
        else:
            print(e)
            return None


def rtm(x, base=5):
    """
    Function for rounding integer value to higher / lower value based on multiple value.
    Funkcija ce predstaviti broj "number" kao visekratnik broja "multiple".
    :param x: Number for rounding.
    :param base: Multiple that will represent the number value.
    :return:
    """
    if x > 100:
        return 100  # Zaštita jer je count matrica za brzine (0, 100)

    return int(base * round(x / base))


    # number = round(number, 0)
    # rest = int(number / multiple)
    # modul = int(number % multiple)
    # if modul <= int(multiple / 2):
    #     return rest * multiple
    # else:
    #     return (rest * multiple) + multiple


def myround(x, base=5):
    return base * round(x/base)


def round_float(number, decimals):
    """
    Rounds decimal number to exact number of decimals.
    :param number: (float) Float number.
    :param decimals: (int) Number of numbers after decimal point
    :return:
    """
    number = float(number)
    out = round(number, decimals)
    return out


def plot_heatmap(data, title, output='show', filename='image.png'):
    """
    Plots heatmap for all speed transitions.
    :param data: 2D numpy array.
    :param states_names: State names (x and y labels).
    :param title: Title for ploting.
    :param output:
    :param filename:
    :return:
    """
    states_names = config.SPEED_LIST

    fig, ax = plt.subplots(figsize=(8, 8))
    img = ax.imshow(data, cmap='cividis', interpolation='none')
    cbar = fig.colorbar(img, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Number of vehicles')

    ax.set_xticks(np.arange(len(states_names)))
    ax.set_yticks(np.arange(len(states_names)))
    ax.set_xticklabels(states_names)
    ax.set_yticklabels(states_names)

    plt.xlabel('Destination speed (km/h)')
    plt.ylabel('Origin speed (km/h)')

    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(states_names)):
        for j in range(len(states_names)):
            ax.text(j, i, data[i, j], ha="center", va="center", color="w")

    ax.set_title(title)
    fig.tight_layout()

    if output == 'show':
        plt.show()
    if output == 'save':
        plt.savefig(filename, bbox_inches='tight')


def two_sigma(x):
    return [x.mean() - 2 * x.std(), x.mean() + 2 * x.std()]


def KL(a, b):
    sum_a = np.sum(a)
    sum_b = np.sum(b)
    a = a / sum_a
    b = b / sum_b
    a[a <= 0] = 0.000000000001
    b[b <= 0] = 0.000000000001
    a = a.astype('float')
    b = b.astype('float')
    return np.sum(np.where(a != 0, a * np.log(a / b), 0))


def IQR(datacolumn, iqr=1.5):
    datacolumn = sorted(datacolumn)
    Q1, Q3 = np.percentile(datacolumn, [25, 75])
    IQR = Q3 - Q1
    lower_range = Q1 - (iqr * IQR)
    upper_range = Q3 + (iqr * IQR)
    return lower_range, upper_range


def plot_graph(x, y, title, output='show', filename='imageG.png'):


    plt.plot(x, y)


    if output == 'show':
        plt.show()
    if output == 'save':
        plt.savefig(filename, bbox_inches='tight')


def plot_graph2(x, y, xlab, ylab, title, output='show', filename='imageG.png'):


    plt.plot(x, y)
    plt.ylabel(ylab)
    plt.xlabel(xlab)
    plt.title(title)

    if output == 'show':
        plt.show()
    if output == 'save':
        plt.savefig(filename, bbox_inches='tight')

def print_exception_msg(e):
    if hasattr(e, 'message'):
        print(e.message)
    else:
        print(e)


def remove_duplicates(list_):
    points_no_duplicates = []
    for br in range(0, len(list_)):
        try:
            r = list_[br]['link_id']
            n_r = list_[br + 1]['link_id']
            if r == n_r:
                continue
            else:
                points_no_duplicates.append(list_[br])
        except:
            points_no_duplicates.append(list_[br])
    return points_no_duplicates


def check_multiple_links():
    """
    Checks if route have same consecutive links with same id.
    Consecutive links means that error occurred while reading the raw data.
    :return:
    """
    routes = open_pickle(config.ROUTES_PKL_NAME)
    print('routes loaded -------------------- \n')
    danger = []
    rc = 0
    for r in routes:
        if rc % 100 == 0:
            print(rc)

        for i in range(0, r.points.shape[0] - 1):
            if r.points.iloc[i]['link_id'] == r.points.iloc[i + 1]['link_id']:
                danger.append(r.route_id)
        rc += 1

    with open('route_ids_duplicate_links.txt', 'w') as f:
        for item in danger:
            f.write("%s\n" % item)


# def mahalanobis_distance(x=None, data=None, cov=None):
#     """Compute the Mahalanobis Distance between each row of x and the data
#     x    : vector or matrix of data with, say, p columns.
#     data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
#     cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
#     """
#     x_minus_mu = x - np.mean(data)
#     if not cov:
#         cov = np.cov(data.T).astype('int')
#     det = np.linalg.det(cov)
#     inv_covmat = np.linalg.inv(cov)
#     left_term = np.dot(x_minus_mu, inv_covmat)
#     mahal = np.dot(left_term, x_minus_mu.T)
#     return mahal.diagonal()


# def mahalanobis_distance(var1, var2):
#     X = np.vstack((var1, var2))
#     m_dist = mahalanobis(var1, var2, np.cov(X))
#     return m_dist


# def mahalanobis_distance(var1, var2):
#     X = np.vstack((var1, var2)).T
#     covariance = np.cov(X).astype('int')
#     inv_cov = np.linalg.inv(covariance)
#     m_dist = mahalanobis(var1, var2, inv_cov)
#     return m_dist


def scale_to(matrix, scale_max):
    # Scaling the speed count matrix to values [0-100]
    max_val = matrix.max()
    factor = scale_max / max_val
    matrix = matrix * factor
    # matrix_ = matrix.astype(np.uint8)
    return matrix


def scale_to_value(matrix, value):
    sum = np.sum(matrix)
    rez = matrix / sum
    rez = scale_to(rez, value)
    return rez


def normalize(matrix, value):
    sum = np.sum(matrix)
    rez = matrix / sum
    rez = rez * value
    return rez.astype('int')


def get_matrix_median(matrices):
    # https://stackoverflow.com/questions/18826422/python-element-wise-means-of-multiple-matrices-with-numpy
    N = int(len(matrices) - 1)
    t = int(len(matrices))
    median_matrix_all = np.median([matrices[t - j] for j in range(1, N + 1)], axis=0)
    return median_matrix_all


# def set_anomaly_and_type(dataframe, top_clusters=3):
#     for i in range(0, top_clusters):
#         dataframe.loc[i, 'anomaly'] = 1
#         a_type = get_anomaly_type(dataframe.loc[i, 'c_data'])
#         dataframe.loc[i, 'anomaly_type'] = a_type
#     return dataframe
#
#
# def get_anomaly_type(matrix):
#     matrix = np.where(matrix < 20, 0, matrix)
#     matrix = np.where(matrix > 0, 1, matrix)
#     # max_val_index = np.where(matrix == np.max(matrix))
#     # row = int(max_val_index[0][0])
#     # column = int(max_val_index[1][0])
#     n_half = int(matrix.shape[0]/2)  # nxn matrix
#
#     n_q1 = 0
#     n_q2 = 0
#     n_q3 = 0
#     n_q4 = 0
#     for row in range(0, matrix.shape[0]):
#         for column in range(0, matrix.shape[1]):
#             if matrix[row, column] == 1:
#                 # First quadrant.
#                 if row < n_half and column >= n_half:
#                     n_q1 += 1
#                 # Second quadrant.
#                 elif row < n_half and column < n_half:
#                     n_q2 += 1
#                 # Third quadrant.
#                 elif row >= n_half and column < n_half:
#                     n_q3 += 1
#                 # Fourth quadrant.
#                 else:
#                     n_q4 += 1
#
#     quadrants = [n_q1, n_q2, n_q3, n_q4]
#     return quadrants.index(max(quadrants)) + 1
