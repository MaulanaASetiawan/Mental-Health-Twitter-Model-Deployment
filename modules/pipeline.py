"""
Pipeline module to initialize the TFX pipeline.
"""
import os
from typing import Text
from absl import logging
from tfx.orchestration import metadata, pipeline


def init_pipeline(
    pipeline_root: Text,
    pipeline_name: str,
    metadata_path: Text,
    components,
) -> pipeline.Pipeline:
    """
    Initialize the pipeline with the given components.
    Args:
        components: List of components to be added to the pipeline.
        pipeline_root: Directory to save the pipeline artifacts.
        pipeline_name: Name of the pipeline.
        metadata_path: Path to the metadata database.
    Returns:
        The TFX pipeline object.
    """

    logging.info(f"Pipeline root set to: {pipeline_root}")

    beam_pipeline_args = [
        "--direct_running_mode=multi_processing",
        f"--temp_location={os.path.join(pipeline_root, 'tmp')}",
    ]

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            metadata_path
        ),
        beam_pipeline_args=beam_pipeline_args,
    )
