####################################################################
AI4EBV: Artificial Intelligence for Essential Biodiversity Variables
####################################################################

This is the code repository for the AI4EBV project. The repository
hosts the ``ai4ebv`` Python package, which relies on the
`pysegcnn <https://gitlab.inf.unibz.it/REMSEN/ccisnow/pysegcnn>`_ package.

Installation
============

Requirements
------------

``ai4ebv`` requires ``Python>=3.7``. The list of dependencies, in addition to `PyTorch <https://pytorch.org/>`_ is defined in ``environment.yml`` and ``requirements.txt``.

Download
---------
You can download ``ai4ebv`` from this repository's
`website <https://gitlab.inf.unibz.it/REMSEN/ai4ebv>`_
or alternatively use ``git`` from terminal:

.. code-block:: bash

    git clone https://gitlab.inf.unibz.it/REMSEN/ai4ebv.git

This creates a copy of the repository in your current directory on the file
system.

Conda package manager
---------------------

To install ``ai4ebv``, we recommend to use the ``conda`` package manager.
You can download ``conda`` `here <https://docs.conda.io/en/latest/miniconda.html>`_.
Once successfully installed ``conda``, we recommend to add ``conda-forge`` as
your default channel:

.. code-block:: bash

    conda config --add channels conda-forge

Conda environment
-----------------

To install ``ai4ebv``, we recommend to create a specific ``conda``
`environment <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_,
by using the provided ``environment.yml`` file. In a terminal, navigate to the
**cloned git repositories root directory** and type:

.. code-block:: bash

    conda env create -f environment.yml

This may take a while. The above command creates a conda environment with all
required dependencies except the `PyTorch <https://pytorch.org/>`_ package. The first line in
``environment.yml`` defines the environment name, in this case ``ai4ebv``.
Activate your environment using:

.. code-block:: bash

    conda activate ai4ebv

Alternatively, use the ``requirements.txt`` file to install the dependencies in an existing conda environment:

.. code-block:: bash

    conda install --file requirements.txt


Install ``pysegcnn``
--------------------

To install the local ``pysegcnn`` package, follow the instructions `here <https://gitlab.inf.unibz.it/REMSEN/ccisnow/pysegcnn/-/blob/master/README.rst>`_.

Install PyTorch
---------------
The installation of ``pytorch`` is heavily dependent on the hardware of your
machine. Therefore, after activating your environment, install the version of
`PyTorch <https://pytorch.org/>`_ that your system supports by following the
official `instructions <https://pytorch.org/get-started/locally/>`_.

If you have to build ``pytorch`` from source, follow this
`guide <https://github.com/pytorch/pytorch#from-source>`_.

To verify the installation, run some sample PyTorch
`code <https://pytorch.org/get-started/locally/#linux-verification>`_.

Install ``ai4ebv`` package
--------------------------
To finally install ``ai4ebv`` run the below command **from this repositories
root directory within the activated ``ai4ebv`` conda environment**:

.. code-block:: bash

    pip install -e .

If successful, you should be able to import ``ai4ebv`` from any Python
interpreter using:

.. code-block:: python

    import ai4ebv

Contact
=======
For further information or ideas for future development please contact:
daniel.frisinghelli@eurac.edu.

License
=======
If not explicitly stated otherwise, this repository is licensed under the
**GNU GENERAL PUBLIC LICENSE v3.0**
(see `LICENSE <https://gitlab.inf.unibz.it/REMSEN/ai4ebv/-/blob/master/LICENSE>`_).

