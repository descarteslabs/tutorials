"""
==================================================
Image uploader
==================================================

Standalone script to create a product (or overwrite an existing one) 
and upload geotiffs into seperate bands or individual scenes

"""

import descarteslabs as dl
import argparse
import rasterio
from datetime import datetime
from collections import namedtuple
import os
import math
from numpy import ma

Band = namedtuple('Band', ['idx', 'dtype', 'description', 'units', 'nodata'])
catalog = dl.Catalog()


def create_product(args):
    # Create a product entry in our Catalog
    try:
        product = catalog.add_product(
            args.product_id,
            title=args.title or 'Image',
            description=args.description or 'Description',
        )
    except Exception as e:
        overwrite = input(
            'That product already exists. [O]verwrite or [A]ppend? [A]: ')
        if overwrite.lower() == 'o':
            catalog.remove_product(
                dl.Auth().namespace + ':' + args.product_id, cascade=True)
            product = catalog.add_product(
                args.product_id,
                title=args.title or 'Image',
                description=args.description or 'Description'
            )
        else:
            product = catalog.get_product(args.product_id, add_namespace=True)

    return product


def create_band(srcfile, filename, product_id):

    src = rasterio.open(filename)

    band_name = filename.split('/')[-1].replace('.tif', '')
    bands = [
        Band(i, dt, d, u, n)
        for i, dt, d, u, n in
        zip(src.indexes,
            src.dtypes,
            src.descriptions,
            src.units,
            src.nodatavals)
    ]

    # Get info from catalog or from file
    for band in bands:

        if len(bands) > 1:
            band_id = f'band_name_Band{band.idx}'
        else:
            band_id = band_name

        band_arr = src.read(band.idx)

        data_range = [
            float(ma.masked_equal(band_arr, band.nodata).min()),
            float(ma.masked_equal(band_arr, band.nodata).max())
        ]

        nodata = band.nodata
        if nodata > 9223372036854775807:
            nodata = None

        catalog.add_band(
            product_id,
            band_id,
            srcfile=srcfile,
            nbits=band_arr.dtype.itemsize*8,
            srcband=band.idx,
            dtype=band.dtype.capitalize(),
            data_range=data_range,
            default_range=data_range,
            type='spectral',
            nodata=nodata
        )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'image_files', type=str, nargs='+', help='Path to GeoTiff files'
    )
    parser.add_argument(
        '--seperate', type=bool, default=False, help='Put files into seperate bands of the product. Otherwise it is assumed that each image is a scene.'
    )
    parser.add_argument(
        '--product_id', type=str, required=True, help='New or existing product ID'
    )

    parser.add_argument(
        '--title', type=str,  help='A title'
    )
    parser.add_argument(
        '--description', type=str,  help='A description'
    )

    args = parser.parse_args()

    product = create_product(args)

    if args.seperate == True:

        for index, filename in enumerate(args.image_files):

            create_band(index, filename, product['data']['id'])

        print(f"Uploading {args.image_files} to product id {args.product_id}")
        print("Please wait...")

        catalog.upload_image(
            args.image_files,
            product['data']['id'],
            multi=True,
            image_id='_'.join(['image_', str(datetime.now().isoformat())]))
    else:
        for index, filename in enumerate(args.image_files):
            create_band(index, filename, product['data']['id'])

        print(f"Uploading {args.image_files} to product id {args.product_id}")
        print("Please wait...")
        for image in args.image_files:
            catalog.upload_image(
                args.image_files,
                product['data']['id']
            )
    print("Done.")
