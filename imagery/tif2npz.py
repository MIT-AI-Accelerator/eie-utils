import sys, os, time
import pandas as pd
import numpy as np
import json

from collections import defaultdict

import fiona
import fiona.transform
import rasterio
import rasterio.mask
import shapely
import shapely.geometry

NLCD_CLASSES = [
    0, 11, 12, 21, 22, 23, 24, 31, 41, 42, 43, 51, 52, 71, 72, 73, 74, 81, 82, 90, 95, 255
]
NLCD_CLASSES_TO_IDX = defaultdict(lambda: 0, {cl:i for i,cl in enumerate(NLCD_CLASSES)})
NLCD_CLASS_IDX = range(len(NLCD_CLASSES))


def humansize(nbytes):
    suffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    i = 0
    while nbytes >= 1024 and i < len(suffixes)-1:
        nbytes /= 1024.
        i += 1
    f = ('%.2f' % nbytes).rstrip('0').rstrip('.')
    return '%s %s' % (f, suffixes[i])

def get_lc_stats(data):
    vals = [1, 2, 3, 4, 5, 6, 15]
    counts = []
    for val in vals:
        counts.append((data==val).sum())
    return np.array(counts)


def get_nlcd_stats(data):
    counts = []
    for val in NLCD_CLASSES:
        counts.append((data==val).sum())
    return np.array(counts)


####Metrics on tile size####
num_tiles = 100
samples_per_tile = 125
sample_size = 1024
num_channels = 3
num_bytes_per_channel = 1
average_tile_size = 7000

print("Number of samples", num_tiles * samples_per_tile)
print("Number of samples_per_tile that will give complete coverage", (average_tile_size/sample_size) * (average_tile_size/sample_size))
print("Expected fraction of each tile that will be sampled", (samples_per_tile * (sample_size*sample_size)) / (average_tile_size*average_tile_size))
print("Size of sampled data", humansize(
    ((6 * num_tiles * samples_per_tile * (sample_size*sample_size) * 11 * 4) + 
    (6 * num_tiles * samples_per_tile * (sample_size*sample_size) * 18 * 4)) / 16
))
#########


def make_dataset_big(fns, state, dataset, base_dir, output_dir):

    patch_fns = []
    patch_metadata = []
    patch_shapes = []
        
    for i, lc_fn in enumerate(fns):
        print(i, len(fns))

        base_fn = "_".join(os.path.basename(lc_fn).split("_")[:-1])
        temp_fns = [
            base_fn + "_naip-new.tif",
            base_fn + "_naip-old.tif",
            base_fn + "_lc.tif",
            base_fn + "_nlcd.tif",
            base_fn + "_landsat-leaf-on.tif",
            base_fn + "_landsat-leaf-off.tif",
            base_fn + "_buildings.tif",
        ]
        
        input_fns = [
            os.path.join(base_dir, "%s_%s_tiles" % (state, dataset), fn)
            for fn in temp_fns
        ]
        
        layer_data = []
        left, bottom, right, top = None, None, None, None
        crs = None
        for fn in input_fns:
            #print("Loading %s" % (fn))
            f = rasterio.open(fn,"r")
            data = f.read()
            left, bottom, right, top = f.bounds
            crs = f.crs.to_string()
            f.close()
            layer_data.append(data)
        
        _, height, width = layer_data[0].shape

        for j in range(samples_per_tile):

            y = np.random.randint(0, height-sample_size)
            x = np.random.randint(0, width-sample_size)

            merged = np.concatenate([
                data[:, y:y+sample_size, x:x+sample_size]
                for data in layer_data
            ])

            lc_string = ','.join(map(str,get_lc_stats(merged[8,:,:])))
            nlcd_string = ','.join(map(str,get_nlcd_stats(merged[9:,:])))
            
                        
            t_left = left + x
            t_right = left + x + sample_size
            t_top = top - y
            t_bottom = top - y - sample_size
            t_geom = shapely.geometry.mapping(shapely.geometry.box(t_left, t_bottom, t_right, t_top, ccw=True))
            t_geom = fiona.transform.transform_geom(crs, 'epsg:4326', t_geom)

            output_fn = "%s-%s-%d.npz" % (
                state,
                base_fn,
                j
            )
            
            #print('Before shape:', merged.shape)
            #print('After shape:',merged[0:3,:,:].shape)
            merged = merged[0:num_channels,:,:]
            np.savez_compressed(os.path.join(output_dir, output_fn), merged[np.newaxis].data)
            patch_fns.append(os.path.join(output_dir, output_fn))
            patch_metadata.append((
                base_fn,
                x, y,
                lc_string,
                nlcd_string
            ))
            patch_shapes.append(json.dumps(t_geom))
    
    return patch_fns, patch_metadata, patch_shapes

state="va"
extended_ds="1m_2014_extended-train"
output_dir="/data2/%s_%s_patches_1024/" % (state, extended_ds)
f = open("/data2/va_1m_2014_extended-train_tiles.csv")
filenames = f.read().strip().split('\n')
fns = []
for fn in filenames[1:]:
	fns.append(fn.split(',')[1])

print(fns[:10])

patch_fns, patch_metadata, patch_shapes = make_dataset_big(
            fns, state, extended_ds, "/data2/", output_dir
        )
