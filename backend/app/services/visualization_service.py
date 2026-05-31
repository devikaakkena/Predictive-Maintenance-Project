import os

class VisualizationService:
    @staticmethod
    def get_static_graphs_list():
        """Returns the relative paths of the pre-generated graphs for rendering."""
        return [
            "graphs/feature_importance.png",
            "graphs/failure_vs_safe.png",
            "graphs/correlation_heatmap.png"
        ]
