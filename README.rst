lathe
=====

Basic machine learning tools for BYU CS478.

requirements
------------

-  `python2.7 or python3.3+ <https://www.python.org/downloads/>`__
-  `pip <https://pip.pypa.io/en/stable/installing/>`__ (*optional*)

installation
------------

::

    pip install lathe

usage
-----

.. code:: python

    import lathe

    args = lathe.parse_args()
    data, targets = lathe.load(args.file, label_size=args.layers[-1])
