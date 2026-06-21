"""agents — six responsibility-isolated coder-agent areas (see agents/README.md).

Each subpackage owns ONE stage and talks to the rest only through `shared.contract`:
detector · orchestrator · classic_cleaner · neural_cleaner · diffusion_cleaner · validator.
"""
