from hivememory.artifact import ReasoningArtifact, Evidence, Conflict

__all__ = ["HiveMemory", "ReasoningArtifact", "Evidence", "Conflict"]


def __getattr__(name):
    if name == "HiveMemory":
        from hivememory.core import HiveMemory

        return HiveMemory
    raise AttributeError(f"module 'hivememory' has no attribute {name}")
