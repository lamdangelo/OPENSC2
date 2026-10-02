"""Shared code of the W7-X publication models (models/w7x).

Modules:
    constants           event-time conventions of the AAB51 quench test
    measurement_data    readers of the AAB51 measurement subset
    controller          step callback driving the winding-pack (WP) runs
    run_wp_model        command line entry point used by every run_model.py
    figure_library      standard figures of a WP run
    publication_figures publication figures of a WP run
    no_network_figures  figure set of the run without hydraulic network
"""
