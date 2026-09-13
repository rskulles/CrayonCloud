"""The LoRA folder: drop a .safetensors adapter in, it appears in the list, and a request can ask for it by name."""

from __future__ import annotations

import platform
from dataclasses import dataclass
from pathlib import Path


def default_loras_dir() -> Path:
    if platform.system() == "Darwin":
        return Path.home() / "Library" / "Application Support" / "CrayonCloud" / "loras"
    return Path.home() / ".config" / "crayoncloud" / "loras"


@dataclass(frozen=True)
class LoraInfo:
    name: str
    file: str
    size: int


class LoraLibrary:
    def __init__(self, folder: Path) -> None:
        self.folder = folder

    def list(self) -> list[LoraInfo]:
        if not self.folder.is_dir():
            return []
        files = sorted(p for p in self.folder.iterdir() if p.is_file() and p.suffix.lower() == ".safetensors")
        return [LoraInfo(p.stem, p.name, p.stat().st_size) for p in files]

    def resolve(self, name: str) -> Path:
        """The file for a name (the stem, case-insensitive). Only files inside the folder count; no paths accepted."""
        wanted = name.strip().lower().removesuffix(".safetensors")
        if not wanted or "/" in wanted or "\\" in wanted or wanted in (".", ".."):
            raise KeyError(name)
        for info in self.list():
            if info.name.lower() == wanted:
                return self.folder / info.file
        raise KeyError(name)
