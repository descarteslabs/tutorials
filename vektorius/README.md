# Getting Started

## Setting up a new virtualenv
Vektorius is only available through the Alpha release of the client and can only be used with Python 3.6 or greater.  To get started you should create a new virtual environment:

```
git clone https://github.com/descarteslabs/tutorials.git
cd tutorials/vektorius
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

And install the latest Alpha client, instructions are [here](https://docs.descarteslabs.com/installation.html#alpha-installation).


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

And install the latest Alpha client, instructions are [here](https://docs.descarteslabs.com/installation.html#alpha-installation).

Note that this will overwrite the default `descarteslabs` install, so proceed with caution!


