"""Shared, importable implementation of the CSI pipeline.

Entry-point scripts add ``<project>/src`` to ``sys.path`` and import from here,
so batch preprocessing, the experiments and ``run_episode.py`` all execute the
exact same ingestion, quality-control, windowing and normalization code.
"""
