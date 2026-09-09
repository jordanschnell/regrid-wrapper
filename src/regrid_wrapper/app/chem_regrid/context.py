from abc import ABC
from enum import StrEnum, unique
from functools import cached_property
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel, Field

T = TypeVar("T", bound="RwBaseModel")


class RwBaseModel(ABC, BaseModel):
    model_config = {"frozen": True}

    @classmethod
    def from_yaml(cls: type[T], data: dict) -> T:
        return cls.model_validate(data)

    @classmethod
    def from_yaml_file(cls: type[T], path: Path) -> T:
        string_data = path.read_text()
        yaml_data = yaml.safe_load(string_data)
        return cls.from_yaml(yaml_data)


@unique
class DatasetName(StrEnum):
    RAVE = "RAVE"
    GRA2PES = "GRA2PES"
    NEMO_RWC = "NEMO_RWC"
    NEMO_ANTHRO = "NEMO_ANTHRO"
    FMC = "FMC"
    PECM = "PECM"
    NARR = "NARR"
    ECOREGION = "ECOREGION"
    FENGSHA_2D = "FENGSHA_2D"
    FENGSHA_2D_Time = "FENGSHA_2D_Time"
    NGFS = "NGFS"
    GOES = "GOES"
    GOES_WEST_AOD = "GOES_WEST_AOD"
    GOES_WEST_ADP = "GOES_WEST_ADP"
    GOES_EAST_AOD = "GOES_EAST_AOD"
    GOES_EAST_ADP = "GOES_EAST_ADP"
    EROD = "EROD"


class ChemRegridContext(RwBaseModel):
    dataset_name: DatasetName
    workdir: Path
    input_dir: Path
    output_dir: Path
    weight_dir: Path
    cycle: str = Field(pattern=r"^\d{10}$")  # Validates YYYYMMDDHH format
    mesh_name: str
    scrip_path: Path | None
    dst_path: Path | None
    ebb_dcycle: int
    fcst_length: int

    @cached_property
    def rw_scrip_path(self) -> Path:
        if self.scrip_path is None:
            return self.workdir / f"mpas_{self.mesh_name}_scrip.nc"
        return self.scrip_path

    @cached_property
    def rw_dst_path(self) -> Path:
        if self.dst_path is None:
            return self.workdir / "init.nc"
        return self.dst_path

    @cached_property
    def rw_desc_stats_out(self) -> Path:
        return self.workdir / f"desc_stats-{self.cycle}.csv"

    def get_weight_path(self, interp_method: str) -> Path:
        weight_path = self.weight_dir / (
            "weights_" + self.dataset_name.value + "-to-" + "mpas_" + self.mesh_name + "_" + interp_method + ".nc"
        )
        return weight_path
