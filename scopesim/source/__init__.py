from . import source_utils
# Importing source_templates here creates an import cycle via
# source_templates -> optics -> effects -> ter_curves_utils -> source_templates.
# Keep source_templates lazily importable as scopesim.source.source_templates.

