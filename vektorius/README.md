# Getting Started
Vektorius is an Alpha product, and requires using the `edge` client. It is _highly_ recommended that you use a virtualenv when installing the `edge` client to ensure that you don't overwrite stable installations of the `descarteslabs` client with an `edge` client.

## Setting up a new virtualenv
Vektorius is only available through the `edge` client and can only be used with Python 3.6 or greater.  To get started you should create a new virtual environment:

```
git clone https://github.com/descarteslabs/tutorials.git
cd tutorials/vektorius
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Then run Jupyter with 

```
jupyter notebook --notebook-dir .
```

## Using Workbench
If you're using [Workbench](https://workbench.descarteslabs.com/) you can clone the repo and install the requirements to get started.

```
git clone https://github.com/descarteslabs/tutorials.git
pip install -I -r tutorials/vektorius/requirements.txt
```

Note that this will overwrite the default `descarteslabs` install with an `edge` client, so proceed with caution! You can revert these changes at any time with `pip install -I "descarteslabs[complete]"`.

## Updating the client

You can update the latest version of the client with `pip install --upgrade -r requiments.txt`.