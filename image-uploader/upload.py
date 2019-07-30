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
from numpy import ma

from datetime import datetime
from collections import namedtuple

Band = namedtuple('Band', ['idx', 'dtype', 'description', 'units', 'nodata'])
catalog = dl.Catalog()

class Uploader(object):
    
    def __init__(self):
        
        parser = argparse.ArgumentParser()
        parser.add_argument(
            'image_files', type=str, nargs='+', help='Path to GeoTiff files'
        )
        parser.add_argument(
            '--seperate', type=bool, default=False, help='Put files into seperate bands of the product. Otherwise it is assumed that each image is a scene.'
        )
        parser.add_argument(
            '--product_id', type=str, required=False, help='New or existing product ID'
        )
        parser.add_argument(
            '--title', type=str,  help='A title'
        )
        parser.add_argument(
            '--description', type=str,  help='A description'
        )
        self.args = parser.parse_args()
        
        self.catalog = dl.Catalog()

    def upload(self):

        self.create_product()
        
        if self.args.seperate == True:
            self.upload_to_bands()
        else:
            self.upload_to_scenes()
        print("Done.")


    def create_product(self):
        # Create a product entry in our Catalog
        product_id = self.args.product_id or self.args.image_files[0].replace(
            '.tif','').replace(' ','_')
        try:
            print(f"Creating new product {product_id}")
            product = catalog.add_product(
                product_id,
                title=self.args.title or product_id,
                description=self.args.description or product_id,
            )
        except dl.exceptions.ClientError:
            self.overwrite = input(
                'That product already exists. [O]verwrite or [A]ppend? [A]: ')
            if self.overwrite.lower() == 'o':
                self.catalog.remove_product(
                    dl.Auth().namespace + ':' + product_id, cascade=True)
                product = self.catalog.add_product(
                    product_id,
                    title=self.args.title or product_id,
                    description=self.args.description or product_id
                )
            else:
                product = catalog.get_product(product_id, add_namespace=True)
        self.product = product
       
    def create_bands(self, srcfile, filename):
        print(f'Inspecting {filename}')
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
                band_id = f'b{band.idx}'
            else:
                band_id = band_name
            print(f"Adding band {self.product['data']['id']}:{band_id}")

            band_arr = src.read(band.idx)

            nodata = band.nodata
            if nodata is not None and nodata > 9223372036854775807: 
                nodata = None
                data_range = [
                    float(ma.masked_equal(band_arr, band.nodata).min()),
                    float(ma.masked_equal(band_arr, band.nodata).max())
                ]
            else: 
                data_range = [
                    float(band_arr.min()),
                    float(band_arr.max())
                ]

            

            print(f"Added band {self.product['data']['id']}:{band_id}")
            catalog.add_band(
                self.product['data']['id'],
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
    
    def upload_to_bands(self):
        if self.overwrite:
            for index, filename in enumerate(self.args.image_files):
                self.create_bands(index, filename)

        print(f"Uploading {self.args.image_files} to product id {self.product['data']['id']}")
        print("Please wait...")

        self.catalog.upload_image(
            self.args.image_files,
            self.product['data']['id'],
            multi=True,
            image_id='_'.join(['image_', str(datetime.now().isoformat())]))
    
    def upload_to_scenes(self):
        for index, filename in enumerate(self.args.image_files):
            self.create_bands(index, filename)

        print(f"Uploading {self.args.image_files} to product id {self.product['data']['id']}")
        print("Please wait...")
        for image in self.args.image_files:
            catalog.upload_image(
                image,
                self.product['data']['id']
        )
if __name__ == '__main__':
    uploader = Uploader()
    uploader.upload()
