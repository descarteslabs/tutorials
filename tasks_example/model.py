"""
Example model as an introduction to DL Tasks. Tasks enable your code to
execute in a cloud environment with high throughput access to imagery and
other resources.

This file is structured as a console application with several commands,
representing a simplified model development lifecycle (train, run, and deploy).
Simply running model.py (i.e., python model.py) will show what's available.

The train step pulls Sentinel-1 GRD imagery over an area, and trains a random
forest classifier to detect open water (as distinguished by the CDL/NLCD in
the U.S.).  This model is then serialized (saved) into a file for further use.

The create product step configures a product in the catalog to store the
classification output.

The run step executes such a model over a desired area locally for quick
iterations during development.

The deploy step executes the model remotely in the Tasks cloud environment,
for effortless scaling over large geographical areas with the help of tiling.

Example usage:
    python model.py train foo-abc
    python model.py create-product foo-abc-output
    python model.py run foo-abc 1024:0:90.0:15:-2:35 user-namespace-hash:foo-abc-output
    python model.py deploy foo-abc user-namespace-hash:foo-abc-output
"""

import click
import os
import sys
import descarteslabs as dl
from descarteslabs.client.services.tasks import as_completed


@click.group()
def cli():
    pass


@cli.command(help="Create a catalog product for the model output")
@click.argument("product_name")
def create_product(product_name):
    catalog = dl.Catalog()

    # Create a product in the catalog. We will upload output rasters from the
    # model to this product for storage and further consumption.
    product = catalog.add_product(
        product_name,
        title="Water Classification Example",
        description="Water Classification Example",
    )

    # Configure a band for our classification output
    catalog.add_band(
        product["data"]["id"],
        name="water",
        type="mask",
        srcband=1,  # 1-based index
        dtype="Byte",
        nbits=8,
        data_range=[0, 255],
    )

    print("Created product", product["data"]["id"])


@cli.command(help="Train model locally")
@click.argument("model_version")
def train(model_version):
    train_aoi = {
        "type": "Polygon",
        "coordinates": [
            [
                [-90.13265132904053, 45.66583901267495],
                [-89.26623076200485, 45.66583901267495],
                [-89.26623076200485, 46.21163054080662],
                [-90.13265132904053, 46.21163054080662],
                [-90.13265132904053, 45.66583901267495],
            ]
        ],
    }

    return train_model(model_version, train_aoi)


@cli.command(help="Run model locally")
@click.argument("model_version")
@click.argument("dltile")
@click.argument("output_product")
def run(model_version, dltile, output_product):
    return run_model(model_version, dltile, output_product)


@cli.command(help="Deploy model using tasks")
@click.argument("model_version")
@click.argument("output_product")
def deploy(model_version, output_product):
    deploy_aoi = {
        "type": "Polygon",
        "coordinates": [
            [
                [-99.24164417538321, 26.138411465362807],
                [-93.37666136803256, 26.138411465362807],
                [-93.37666136803256, 31.060649553995205],
                [-99.24164417538321, 31.060649553995205],
                [-99.24164417538321, 26.138411465362807],
            ]
        ],
    }

    # Make sure the output product exists
    try:
        dl.Catalog().get_product(output_product)
    except dl.client.exceptions.NotFoundError:
        print("Output product {} does not exist".format(output_product))
        return

    # Decompose our AOI into 1024x1024 pixel tiles at 90m resolution in UTM
    tiles = dl.scenes.DLTile.from_shape(
        deploy_aoi, resolution=90.0, tilesize=1024, pad=0
    )

    # Register our prediction function in the Tasks environment.
    #
    # We specify the resource requirements per worker (1 CPU & 2GB of RAM),
    # the environment (container with Python 3.7), and any extra PyPI
    # requirements (descarteslabs client and scikit-learn).
    tasks = dl.Tasks()
    run_model_remotely = tasks.create_function(
        run_model,
        name="example water model deployment",
        image="us.gcr.io/dl-ci-cd/images/tasks/public/py3.7/default:v2019.05.29",
        cpu=1.0,
        memory="2Gi",
        requirements=["descarteslabs[complete]==0.19.0", "scikit-learn==0.21.1"],
    )

    # Create a list with arguments of each invocation of run_model
    task_arguments = [(model_version, dltile.key, output_product) for dltile in tiles]

    results = run_model_remotely.map(*zip(*task_arguments))
    print(
        "Submitted {} tasks to task group {}...".format(
            len(tiles), run_model_remotely.group_id
        )
    )

    # Iterate through task results as they complete.
    #
    # If some of the tasks failed, we will print the console output and the
    # arguments of that invocation.
    #
    # Note that this is for informational purposes only, and the tasks will
    # continue running if the script is interrupted at this point. You can use
    # https://monitor.descarteslabs.com/ to see the status of all running
    # task groups.
    for i, task in enumerate(as_completed(results, show_progress=False)):
        percent_complete = 100.0 * i / len(results)
        print(
            "Progress update: {} completed out of {} ({:.2f}%) - last task took {:.2f}sec to {}".format(
                i + 1,
                len(results),
                percent_complete,
                task.runtime,
                "succeed" if task.is_success else "fail",
            )
        )

        if not task.is_success:
            print(
                "\nTASK FAILURE with arguments {}:\n{}".format(
                    task.args, task.log.decode()
                )
            )

    # Clean up the task group
    tasks.delete_group_by_id(run_model_remotely.group_id)


def train_model(model_version, aoi_geom):
    # Import everything that's needed for this function. If this function runs
    # in the tasks environment, it won't have access to anything outside of
    # the function, including the imports at the top.
    import descarteslabs as dl
    import numpy as np
    import pickle
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix

    # Find the input sentinel-1 GRD scenes in our AOI
    training_scenes, ctx = dl.scenes.search(
        aoi_geom,
        products=["sentinel-1:GRD"],
        start_datetime="2019-02-01",
        end_datetime="2019-05-01",
        # Let's filter the imagery to only include scenes on the ascending
        # part of the orbit. Unfortunately "pass" is a keyword in Python, and
        # we can't use dl.properties.pass here.
        query=getattr(dl.properties, "pass") == "ASCENDING",
    )
    print("Training scenes:", training_scenes)

    # For demonstration purposes, use a lower resolution to speed things up.
    lowres_ctx = ctx.assign(resolution=90)

    # Calling stack on a scene collection will retrieve our desired imagery in
    # the VV and VH bands. Settings bands_axis to -1 will store the bands as
    # the last dimension in the array, i.e., with the shape (scene, y, x, band)
    feature_stack = training_scenes.stack("vv vh", lowres_ctx, bands_axis=-1)
    print("Input raster dimensions:", feature_stack.shape)

    # Fetch the corresponding raster data from 2018's CDL. We will use it as
    # the truth to train our model.
    cdl_scene, _ = dl.scenes.Scene.from_id("usda:cdl:v1:meta_2018_30m_cdls_v1")
    cdl_data = cdl_scene.ndarray("class", lowres_ctx, bands_axis=-1)
    print("Ground truth raster dimensions:", cdl_data.shape)

    # Open water is categorized as 111 in the CDL raster. Let's mask it out to
    # use as training labels.
    CDL_OPEN_WATER = 111
    ground_truth = cdl_data == CDL_OPEN_WATER

    # Duplicate the training labels for every input scene.
    ground_truth = np.repeat(ground_truth[np.newaxis, ...], len(feature_stack), axis=0)

    # scikit-learn expects the array to be (n_samples, n_features)
    input_features = feature_stack.reshape(-1, 2)

    # Extract a random 20% of the data as a test set. For real models, one
    # might want to use better techniques such as k-fold cross-validation.
    X_train, X_test, y_train, y_test = train_test_split(
        input_features, ground_truth.ravel(), test_size=0.20
    )

    # Train the random forest and print out some statistics
    classifier = RandomForestClassifier(
        n_estimators=10, class_weight="balanced", verbose=2
    )
    classifier.fit(X_train, y_train)

    model_score = classifier.score(X_test, y_test)
    print("Model score: {}".format(model_score))

    print("Classification report:")
    y_test_predict = classifier.predict(X_test)
    print(classification_report(y_test.data, y_test_predict))

    # Save the trained model as a file and upload to DL storage for deployment
    # through tasks.
    output_filename = "water_model_{}".format(model_version)

    print("Saving trained model to {}...".format(output_filename))
    with open(output_filename, "wb") as f:
        pickle.dump(classifier, f)

    print("Uploading model to DL storage...")
    dl.Storage().set_file(output_filename, output_filename)


def run_model(model_version, dltile_key, output_product_id):
    import descarteslabs as dl
    import numpy as np
    import pickle
    import os

    # Load the model, optionally downloading it from DL storage
    model_filename = "water_model_{}".format(model_version)
    if not os.path.exists(model_filename):
        try:
            dl.Storage().get_file(model_filename, model_filename)
        except dl.client.exceptions.NotFoundError:
            print("Could not find {} in DL storage".format(model_filename))
            return

    with open(model_filename, "rb") as f:
        classifier = pickle.load(f)

    # Find input scenes for the dltile
    dltile = dl.scenes.DLTile.from_key(dltile_key)
    input_scenes, ctx = dl.scenes.search(
        dltile,
        products=["sentinel-1:GRD"],
        start_datetime="2019-03-01",
        end_datetime="2019-05-01",
        query=getattr(dl.properties, "pass") == "ASCENDING",
    )
    print("Input scenes:", input_scenes)

    input_stack = input_scenes.stack("vv vh", ctx, bands_axis=-1)

    # Predict
    water_prediction = classifier.predict(input_stack.reshape(-1, 2))
    water_prediction = water_prediction.reshape(input_stack[:, :, :, 0].shape)

    # Aggregate the predictions into a single raster for the tile
    composite_prediction = water_prediction.mean(axis=0)

    # Discretize our prediction from 0.0-1.0 to 0-255
    composite_prediction = (composite_prediction * 255).astype(np.uint8)
    print("Uploading output to the catalog with shape", composite_prediction.shape)

    # Upload the result back to catalog
    upload_id = dl.Catalog().upload_ndarray(
        composite_prediction,
        output_product_id,
        dltile.key,
        geotrans=ctx.geotrans,
        proj4=ctx.proj4,
    )

    print("Upload task id:", upload_id)


if __name__ == "__main__":
    cli()
