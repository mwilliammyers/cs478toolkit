.. lathe documentation master file, created by
   sphinx-quickstart on Fri Mar 24 18:32:26 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


`lathe <https://github.com/mwilliammyers/lathe/>`__
=====

Basic machine learning tools for BYU CS478.

.. image:: ./images/lathe.gif
  :align: right

contents
--------

  * :ref:`genindex`
  * :ref:`modindex`
  * :ref:`search`

.. toctree::
   :maxdepth: 4
   :caption: contents:

  modules

requirements
------------

-  `python2.7 <https://www.python.org/downloads/>`__ or `python3.3+ <https://www.python.org/downloads/>`__
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
   data, targets = lathe.load(args.file, label_size=1)


documentation
-------------

http://lathe.readthedocs.io
