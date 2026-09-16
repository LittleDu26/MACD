#! /usr/bin/env python

import math
import pickle
from pathlib import Path
from sklearn.cluster import KMeans
import numpy as np
import sys,os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.collections import PolyCollection
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon
from sympy.geometry import Point
from sympy.geometry import Polygon as SymPoly
from pyME.map_elites import common as cm

#path info
curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(curr_dir, '../..')

def saved_data_path(*parts):
    saved_data_dir = os.environ.get("EVOGYM_GASH_SAVED_DATA", "saved_data")
    return os.path.join(root_dir, saved_data_dir, *parts)


default_params = \
    {
        # more of this -> higher-quality CVT
        "cvt_samples": 25000,
        # we evaluate in batches to paralleliez
        "batch_size": 100,
        # proportion of niches to be filled before starting
        "random_init": 0.1,
        # batch for random initialization
        "random_init_batch": 100,
        # when to write results (one generation = one batch)
        "dump_period": 10000,
        # do we use several cores?
        "parallel": True,
        # do we cache the result of CVT and reuse?
        "cvt_use_cache": True,
        # min/max of parameters
        "min": 0,
        "max": 1,
        # only useful if you use the 'iso_dd' variation operator
        "iso_sigma": 0.01,
        "line_sigma": 0.2
    }

class Species:
    def __init__(self, x, desc, fitness, centroid=None):
        self.x = x
        self.desc = desc
        self.fitness = fitness
        self.centroid = centroid


def polynomial_mutation(x):
    '''
    Cf Deb 2001, p 124 ; param: eta_m
    '''
    y = x.copy()
    eta_m = 5.0;
    r = np.random.random(size=len(x))
    for i in range(0, len(x)):
        if r[i] < 0.5:
            delta_i = math.pow(2.0 * r[i], 1.0 / (eta_m + 1.0)) - 1.0
        else:
            delta_i = 1 - math.pow(2.0 * (1.0 - r[i]), 1.0 / (eta_m + 1.0))
        y[i] += delta_i
    return y

def sbx(x, y, params):
    '''
    SBX (cf Deb 2001, p 113) Simulated Binary Crossover

    A large value ef eta gives a higher probablitity for
    creating a `near-parent' solutions and a small value allows
    distant solutions to be selected as offspring.
    '''
    eta = 10.0
    xl = params['min']
    xu = params['max']
    z = x.copy()
    r1 = np.random.random(size=len(x))
    r2 = np.random.random(size=len(x))

    for i in range(0, len(x)):
        if abs(x[i] - y[i]) > 1e-15:
            x1 = min(x[i], y[i])
            x2 = max(x[i], y[i])

            beta = 1.0 + (2.0 * (x1 - xl) / (x2 - x1))
            alpha = 2.0 - beta ** -(eta + 1)
            rand = r1[i]
            if rand <= 1.0 / alpha:
                beta_q = (rand * alpha) ** (1.0 / (eta + 1))
            else:
                beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))

            c1 = 0.5 * (x1 + x2 - beta_q * (x2 - x1))

            beta = 1.0 + (2.0 * (xu - x2) / (x2 - x1))
            alpha = 2.0 - beta ** -(eta + 1)
            if rand <= 1.0 / alpha:
                beta_q = (rand * alpha) ** (1.0 / (eta + 1))
            else:
                beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
            c2 = 0.5 * (x1 + x2 + beta_q * (x2 - x1))

            c1 = min(max(c1, xl), xu)
            c2 = min(max(c2, xl), xu)

            if r2[i] <= 0.5:
                z[i] = c2
            else:
                z[i] = c1
    return z


def iso_dd(x, y, params):
    '''
    Iso+Line
    Ref:
    Vassiliades V, Mouret JB. Discovering the elite hypervolume by leveraging interspecies correlation.
    GECCO 2018
    '''
    assert(x.shape == y.shape)
    p_max = np.array(params["max"])
    p_min = np.array(params["min"])
    a = np.random.normal(0, params['iso_sigma'], size=len(x))
    b = np.random.normal(0, params['line_sigma'])
    norm = np.linalg.norm(x - y)
    z = x.copy() + a + b * (x - y)
    return np.clip(z, p_min, p_max)


def variation(x, z, params):
    assert(x.shape == z.shape)
    y = sbx(x, z, params)
    return y

def __centroids_filename(k, dim):
    filename='centroids_' + str(k) + '_' + str(dim) + '.dat'
    pathname=os.path.join(curr_dir,'voronoi_files',filename)
    return pathname


def __write_centroids(centroids):
    k = centroids.shape[0]
    dim = centroids.shape[1]
    filename = __centroids_filename(k, dim)
    with open(filename, 'w') as f:
        for p in centroids:
            for item in p:
                f.write(str(item) + ' ')
            f.write('\n')


def cvt(k, dim, samples, cvt_use_cache=True):
    # check if we have cached values
    fname = __centroids_filename(k, dim)
    if cvt_use_cache:
        if Path(fname).is_file():
            print("WARNING: using cached CVT:", fname)
            return np.loadtxt(fname)
    # otherwise, compute cvt
    print("Computing CVT (this can take a while...):", fname)

    x = np.random.rand(samples, dim)
    k_means = KMeans(init='k-means++', n_clusters=k,
                     n_init=1, verbose=0)#,algorithm="full")
    k_means.fit(x)
    __write_centroids(k_means.cluster_centers_)

    return k_means.cluster_centers_


def make_hashable(array):
    return tuple(map(float, array))


def parallel_eval(evaluate_function, to_evaluate, pool, params):
    if params['parallel'] == True:
        s_list = pool.map(evaluate_function, to_evaluate)
    else:
        s_list = map(evaluate_function, to_evaluate)
    return list(s_list)

def calc_desc(body):
    voxel_count=0
    rigid_count=0
    for x in body:
        for xx in x:
            if xx>0:
                voxel_count+=1
            if xx==1:
                rigid_count+=1
    return rigid_count/(voxel_count*1.0),voxel_count/(body.shape[0]*body.shape[1])


        

# format: fitness, centroid, desc, genome \n
# fitness, centroid, desc and x are vectors
def __save_archive(archive, gen):
    def write_array(a, f):
        for i in a:
            f.write(str(i) + ' ')
    filename = 'archive_' + str(gen) + '.dat'
    with open(filename, 'w') as f:
        for k in archive.values():
            f.write(str(k.fitness) + ' ')
            write_array(k.centroid, f)
            write_array(k.desc, f)
            write_array(k.x, f)
            f.write("\n")


def region_to_centroid(vertices,centroids):
    x=[]
    y=[]
    for vertice in vertices:
        x.append(vertice[0])
        y.append(vertice[1])
    points=[]
    for x_i,y_i in zip(x,y):
        points.append(Point(x_i,y_i))
    poly=SymPoly(*points)
    for centroid in centroids:
        centroid_Point=Point(centroid[0],centroid[1])
        if poly.encloses_point(centroid_Point):
            return centroid
    print("no centroid is founded")
    exit(1)

def join_hash(target):
    hash=''.join(list(map(str,target)))
    return hash

def map_regions_points(vor,region_centroid_dict:dict,regions,centroids):
    for region in regions:
        hash=join_hash(region)
        region_centroid_dict[hash]= region_to_centroid(vor.vertices[region],centroids)

def color_list(score,max_score):
    if score==0:
        return [1.0,1.0,1.0]
    yellow=50
    blue=240
    h_range=360
    scaled=(1-(score/max_score))*(blue-yellow)/h_range
    return colors.hsv_to_rgb([(yellow/h_range)+scaled,1.0,1.0])

def __region_centroids_filename(k):
    filename='region_centroid_dict_' + str(k) + '.pkl'
    pathname=os.path.join(curr_dir,'voronoi_files',filename)
    return pathname

def __vor_object_filename(k):
    filename='vor_obj_'+ str(k) +'.pkl'
    pathname=os.path.join(curr_dir,'voronoi_files',filename)
    return pathname

def __write_region_centroids(n_niches,rc_dict):
    filename = __region_centroids_filename(n_niches)
    with open(filename, 'wb') as f:
        pickle.dump(rc_dict,f)

def __write_vor_object(n_niches,vor):
    filename=__vor_object_filename(n_niches)
    with open(filename, 'wb') as f:
        pickle.dump(vor,f)

def bounded_voronoi(bnd, pnts,centroid_score_dict,max_score,min_score,n_niches,experiment_name,generation):
    """
    有界なボロノイ図を計算・描画する関数．
    """

    # すべての母点のボロノイ領域を有界にするために，ダミー母点を3個追加
    gn_pnts = np.concatenate([pnts, np.array([[100, 100], [100, -100], [-100, 0]])])
    vor_fname = __vor_object_filename(n_niches)
    is_vor_exist=Path(vor_fname).is_file()
    if is_vor_exist:
        print("WARNING: using cached VOR:", vor_fname)
        with open(vor_fname,'rb') as f:
            vor=pickle.load(f)
    else:
        # ボロノイ図の計算
        vor = Voronoi(gn_pnts,qhull_options="Qc")
        __write_vor_object(n_niches,vor)

    
    
    reg_fname = __region_centroids_filename(n_niches)
    if is_vor_exist and Path(reg_fname).is_file():
        print("WARNING: using cached region centroid dict:", reg_fname)
        with open(reg_fname,'rb') as f:
            region_centroid_dict = pickle.load(f)
    else:
        region_centroid_dict={}
        map_regions_points(vor,region_centroid_dict,[r for r in vor.regions if -1 not in r and r],vor.points)
        __write_region_centroids(n_niches,region_centroid_dict)

    # 分割する領域をPolygonに
    bnd_poly = Polygon(bnd)

    # 各ボロノイ領域をしまうリスト
    vor_polys = []

    # ダミー以外の母点についての繰り返し
    for i in range(len(gn_pnts) - 3):

        # 閉空間を考慮しないボロノイ領域
        vor_poly = [vor.vertices[v] for v in vor.regions[vor.point_region[i]]]
        # 分割する領域をボロノイ領域の共通部分を計算
        i_cell = bnd_poly.intersection(Polygon(vor_poly))

        # 閉空間を考慮したボロノイ領域の頂点座標を格納
        vor_polys.append(list(i_cell.exterior.coords[:-1]))


    # ボロノイ図の描画
    fig = plt.figure(figsize=(12, 11))
    ax = fig.add_subplot(111)
    
    # 母点
    ax.scatter(pnts[:,0], pnts[:,1])

    # ボロノイ領域
    poly_vor = PolyCollection(vor_polys, edgecolor="black",
                              facecolors="None", linewidth = 1.0)
    ax.add_collection(poly_vor)
   
    
    for i,region in enumerate([r for r in vor.regions if -1 not in r and r]):
        centroid=region_centroid_dict[join_hash(region)]
        score=centroid_score_dict[join_hash(centroid)]
        
        ax.fill(vor.vertices[region][:, 0],
                vor.vertices[region][:, 1],
                color=color_list(score,max_score))


    xmin = np.min(bnd[:,0])
    xmax = np.max(bnd[:,0])
    ymin = np.min(bnd[:,1])
    ymax = np.max(bnd[:,1])

    ax.set_xlim(xmin-0.1, xmax+0.1)
    ax.set_ylim(ymin-0.1, ymax+0.1)
    ax.set_aspect('equal')

    plt.title("max:"+str(max_score)+" min:"+str(min_score))
    path=saved_data_path(experiment_name, "generation_" + str(generation), "mapping_cell.pdf")
    plt.savefig(path)



def draw_voronoi_map(n_niches,experiment_name,generation):
    #read all centroid
   
    with open( __centroids_filename(n_niches, 2)) as f:
        lines=[line.split(' ') for line in f.readlines()]
        centroids=np.array(lines)[:,:2]
        centroids=centroids.astype("float")

    path=saved_data_path(experiment_name, "generation_" + str(generation), "centroid_score.txt")
    

    centroid_score_dict={}
    max_score=0
    min_score=1e8
    for centroid in centroids:
        centroid_score_dict[join_hash(centroid.tolist())]=0
    
    with open(path) as f:
        lines=[line.split('\t\t') for line in f.readlines()]
        centroids_gene_exist=np.array(lines)[:,1:3]
        centroids_gene_exist=centroids_gene_exist.astype("float")
        scores=np.array(lines)[:,3:].T[0]
        for centroid,score in zip(centroids_gene_exist,scores):
            score_float=float(score.split('\n')[0])
            score_float=score_float if score_float > 0 else 0
            centroid_score_dict[join_hash(centroid.tolist())] =score_float
            max_score=score_float if score_float > max_score else max_score
            min_score=score_float if score_float < min_score else min_score
    
    bnd = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

    # ボロノイ図の計算・描画
    bounded_voronoi(bnd, centroids,centroid_score_dict,max_score,min_score,n_niches,experiment_name,generation)
    
def save_centroid_and_map(root_dir,experiment_name,generation,archive,n_niches):
    temp_path = saved_data_path(experiment_name, "generation_" + str(generation), "centroid_score.txt")
    f = open(temp_path, "w")

    out = ""
    for key in archive:
        out += str(archive[key].label) + "\t\t" + str(key[0]) + "\t\t" + str(key[1]) + "\t\t" + str(archive[key].fitness) + "\n"
    f.write(out)
    f.close()
    draw_voronoi_map(n_niches,experiment_name,generation)
