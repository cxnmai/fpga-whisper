from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .types import BackendKind, PartitionPreset

MODEL_HF_REPO = "distil-whisper/distil-small.en"
MODEL_CT2_ALIAS = "distil-small.en"
MODEL_CT2_CACHE_REPO_DIR = "models--Systran--faster-distil-whisper-small.en"


@dataclass(slots=True)
class AppConfig:
    project_root: Path = field(default_factory=lambda: Path("."))
    worker_module: str = "scripts.ct2_worker"
    reference_python: str = "python3"
    reference_exporter_module: str = "scripts.export_reference_activation"
    reference_cache_dir: Path = field(
        default_factory=lambda: Path("artifacts/reference")
    )
    reference_export_positions: int = 4
    fpga_sim_io_dir: Path = field(default_factory=lambda: Path("fpga/tmp"))
    default_backend: BackendKind = BackendKind.CT2_PYTHON
    default_partition: PartitionPreset = PartitionPreset.HYBRID

    def resolve_project_path(self, relative_path: str | Path) -> Path:
        return self.project_root / Path(relative_path)

    @property
    def uv_cache_dir(self) -> Path:
        return self.resolve_project_path(".uv-cache")

    @property
    def worker_script_path(self) -> Path:
        parts = self.worker_module.split(".")
        return self.resolve_project_path(
            Path("src").joinpath(*parts).with_suffix(".py")
        )

    @property
    def reference_exporter_script_path(self) -> Path:
        parts = self.reference_exporter_module.split(".")
        return self.resolve_project_path(
            Path("src").joinpath(*parts).with_suffix(".py")
        )

    @property
    def resolved_reference_cache_dir(self) -> Path:
        return self.resolve_project_path(self.reference_cache_dir)

    @property
    def resolved_fpga_sim_io_dir(self) -> Path:
        return self.resolve_project_path(self.fpga_sim_io_dir)

    def sample_request(self) -> dict[str, object]:
        return {
            "audio_path": Path("samples/silence.wav"),
            "backend": self.default_backend,
            "partition": self.default_partition,
            "initial_prompt": None,
        }

    def model_snapshot_dir(self) -> Path:
        repo_dir = self.resolve_project_path(Path("models") / MODEL_CT2_CACHE_REPO_DIR)
        revision_path = repo_dir / "refs" / "main"
        revision = revision_path.read_text(encoding="utf-8").strip()
        if not revision:
            raise RuntimeError(
                f"Failed to read baked model revision from {revision_path}"
            )
        return repo_dir / "snapshots" / revision

    def model_bin_path(self) -> Path:
        return self.model_snapshot_dir() / "model.bin"

    def reference_activation_cache_path(self, audio_path: str | Path) -> Path:
        audio_path = Path(audio_path)
        stem = audio_path.stem or "sample"
        filename = (
            f"encoder_layer0_ffn_inputs_{stem}_p{self.reference_export_positions}.json"
        )
        return self.resolved_reference_cache_dir / filename


def default_config(project_root: str | Path = ".") -> AppConfig:
    return AppConfig(project_root=Path(project_root))
