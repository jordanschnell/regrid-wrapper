# mypy: ignore-errors

import sys
import glob
from abc import abstractmethod, ABC
from datetime import datetime, timezone, timedelta
from functools import cached_property
from pathlib import Path
from typing import Literal, Iterable, Any, Union

import os

import esmpy
import numpy as np
import xarray as xr
import pandas as pd
from pydantic import BaseModel

from regrid_wrapper.app.chem_regrid.context import ChemRegridContext
from regrid_wrapper.context.comm import COMM, reconcile_bounds
from regrid_wrapper.context.logging import LOGGER
from regrid_wrapper.esmpy.field_wrapper import (
    GridSpec,
    NcToGrid,
    NcToField,
    FieldWrapper,
    GridWrapper,
    open_nc,
    Dimension,
    DimensionCollection,
    set_variable_data,
    HasNcAttrsType,
    copy_nc_variable, load_variable_data, MeshWrapper,
)

_LOGGER = LOGGER.getChild("mpas-regrid")

NGFS_OUTPUT_NAMES = {
    "FRE": "fre_in",
    "FRP_MEAN": "frp_in",
    "PM25": "e_bb_in_smoke_fine",
}

# Try to find the latest RAVE file available up to max_lookback_hours before target_time_str
# to avoid setting zeroes when a particular hour file is missing.
def find_latest_rave_file(input_dir, target_time_str, ebb_dcycle, dataset_name, max_lookback_hours=24, **kwargs):
    """Return list of files for the latest time <= target_time_str."""
    fmt = "%Y%m%d%H"  #RAVE
    fmt2= "%Y%j%H"  # GOES
    target_time = datetime.strptime(target_time_str, fmt)

    input_dir_str = str(input_dir)

    for h in range(max_lookback_hours + 1):
        if ebb_dcycle == -1 or ebb_dcycle == 2:
           this_time = target_time - timedelta(hours=h)
        elif ebb_dcycle == 1:
           this_time = target_time + timedelta(hours=h)
        else:
           _LOGGER.warning("unrecognized ebb_dcycle, reverting to same-day, ebb_dcycle = 1")
           this_time = target_time + timedelta(hours=h)

        if dataset_name == "RAVE":
           this_str = this_time.strftime(fmt)
           paths = glob.glob(input_dir_str + "/RAVE-HrlyEmiss-3km_v2r0_blend_s"+this_str+"*")
        elif "GOES" in dataset_name:
           valid_args = False
           if 'sat_num' in kwargs.keys():
              SAT_NUM = kwargs['sat_num']
              valid_args = True
           if 'product' in kwargs.keys():
              PRODUCT_CODE = kwargs['product']
              valid_args = True
           if len(kwargs) > 2:
              print("More than 2 optional keyward arguments were passed into find_latest_rave_file. Some will not be used. Check syntax.")
              for key,item in kwargs.items():
                 print(key,item)
           if not valid_args:
              raise KeyError("Incorrect keyward arguments to find_latest_rave_file. If dataset_name = 'GOES', you must provide additional kwargs on the satellite number and product")
           this_str = this_time.strftime(fmt2)
           paths = glob.glob(f"{input_dir_str}/OR_ABI-L2-{PRODUCT_CODE}C-M6_G{SAT_NUM}_s"+this_str+"*")
        if paths:
            if h > 0:
                print(f"Missing {dataset_name} file for {target_time_str}, using {this_str} instead")
            return paths
    # nothing found within lookback window
    return []
#
def create_ngfs_sparse_mesh(lat_1d, lon_1d, resolution=0.01):
    """
    Creates an esmpy.Mesh dynamically from 1-D point source data.
    Calculates the 4 corners of a square cell of size `resolution`
    around each center point in memory.
    This is the best approach since NGFS data are point-source (1-D),
    but we rarely have more than 1000 fires in the domain, so we
    can afford to keep this in memory instead of creating a file.
    """

    num_cells = len(lat_1d)
    if num_cells == 0:
        return None

    num_nodes = num_cells * 4
    d = resolution / 2.0

    node_lons = np.column_stack([
        lon_1d - d, lon_1d + d, lon_1d + d, lon_1d - d
    ]).flatten()

    node_lats = np.column_stack([
        lat_1d - d, lat_1d - d, lat_1d + d, lat_1d + d
    ]).flatten()

    node_coords = np.empty(num_nodes * 2, dtype=np.float64)
    node_coords[0::2] = node_lons
    node_coords[1::2] = node_lats

    node_ids = np.arange(1, num_nodes + 1, dtype=np.int32)
    node_owners = np.full(num_nodes, COMM.rank, dtype=np.int32)

    element_ids = np.arange(1, num_cells + 1, dtype=np.int32)
    element_types = np.full(num_cells, esmpy.MeshElemType.QUAD, dtype=np.int32)

    # CRITICAL FIX: esmpy expects 0-based indexing for connectivity!
    element_conn = np.arange(0, num_nodes, dtype=np.int32)

    # Explicitly set spherical coordinates
    mesh = esmpy.Mesh(parametric_dim=2, spatial_dim=2, coord_sys=esmpy.CoordSys.SPH_DEG)

    mesh.add_nodes(
        node_count=num_nodes,
        node_ids=node_ids,
        node_coords=node_coords,
        node_owners=node_owners
    )

    mesh.add_elements(
        element_count=num_cells,
        element_ids=element_ids,
        element_types=element_types,
        element_conn=element_conn
    )

    return mesh
#
class AbstractRaveField(ABC, BaseModel):
    name: str
    attrs: dict[str, Any]
    fill_value: float
    dtype: Any
    num_cells: int
    level_out_name: str
    level_out_size: int
    time_size: int

    @cached_property
    def time_dimension(self) -> Dimension:
        return Dimension(
            name=("Time",),
            size=self.time_size,
            lower=0,
            upper=self.time_size,
            staggerloc=esmpy.StaggerLoc.CENTER,
            coordinate_type="time",
        )

    @cached_property
    def nklevel_dimension(self) -> Dimension:
        return Dimension(
            name=(self.level_out_name,),
            size=self.level_out_size,
            lower=0,
            upper=self.level_out_size,
            staggerloc=esmpy.StaggerLoc.CENTER,
            coordinate_type="level",
        )

    def create_ncells_dimension(self, bounds: tuple[int, int]) -> Dimension:
        return Dimension(
            name=("nCells",),
            size=self.num_cells,  # 225636, #130333,  # tdk: pull from origin,
            lower=bounds[0],
            upper=bounds[1],
            staggerloc=esmpy.MeshLoc.ELEMENT,
            coordinate_type="cell",
        )

    @abstractmethod
    def create_dimension_collection(
        self, ncells_bounds: tuple[int, int]
    ) -> DimensionCollection:
        ...

    @abstractmethod
    def reshape_field_data(self, target: np.ndarray) -> np.ndarray:
        ...


class RaveField2d(AbstractRaveField):

    def create_dimension_collection(
        self, ncells_bounds: tuple[int, int]
    ) -> DimensionCollection:
        return DimensionCollection(
            value=(self.create_ncells_dimension(ncells_bounds),)
        )

    def reshape_field_data(self, target: np.ndarray) -> np.ndarray:
        return target.reshape(-1)


class RaveField2d_plusTime(AbstractRaveField):

    def create_dimension_collection(
        self, ncells_bounds: tuple[int, int]
    ) -> DimensionCollection:
        return DimensionCollection(
            value=(self.time_dimension, self.create_ncells_dimension(ncells_bounds))
        )

    def reshape_field_data(self, target: np.ndarray) -> np.ndarray:
        return target.reshape(self.time_size, -1)

class RaveField3d(AbstractRaveField):

    def create_dimension_collection(
        self, ncells_bounds: tuple[int, int]
    ) -> DimensionCollection:
        return DimensionCollection(
            value=(
                self.create_ncells_dimension(ncells_bounds),
                self.nklevel_dimension,
            )
        )

    def reshape_field_data(self, target: np.ndarray) -> np.ndarray:
        return target.reshape(-1, self.level_out_size)

class RaveField3d_plusTime(AbstractRaveField):

    def create_dimension_collection(
        self, ncells_bounds: tuple[int, int]
    ) -> DimensionCollection:
        return DimensionCollection(
            value=(
                self.time_dimension,
                self.create_ncells_dimension(ncells_bounds),
                self.nklevel_dimension,
            )
        )
    def reshape_field_data(self, target: np.ndarray) -> np.ndarray:
        return target.reshape(self.time_size, -1, self.level_out_size)

class RaveToMpasRegridContext(BaseModel):
    dataset_name: str
    workdir: Path
    src_path: Path
    dst_path: Path
    new_dst_path: Path
    desc_stats_out: Path
    weight_path: Path
    InterpMethod: str
    scrip_path: Path
    num_cells: int
    mesh_name: str
    field_names: tuple
    x_center: str
    y_center: str
    x_dim: str
    y_dim: str
    x_corner: Union[str, None]
    y_corner: Union[str, None]
    x_corner_dim: Union[str, None]
    y_corner_dim: Union[str, None]
    level_in_name: str
    # level_in_size: int
    level_out_name: str
    level_out_size: int
    time_name: str
    time_size: int
    # InterpMask: float

    rank: int = COMM.rank

    @cached_property
    def rave_fields(self) -> tuple[AbstractRaveField, ...]:
        rave_fields = []
        with open_nc(self.src_path, mode="r") as ds:
            for field_name in self.field_names:
                read_name = field_name
                if self.dataset_name == "NGFS" and field_name == "PM25":
                    read_name = "EMIS_PM25"
                field_level_out_size = self.level_out_size

                if read_name not in ds.variables:
                    raise KeyError(
                        f"Source variable '{read_name}' not found for field '{field_name}' in {self.src_path}"
                    )
                var = ds.variables[read_name]
                init_data = {
                    "name": field_name,
                    "attrs": self._get_nc_attrs_(var),
                    "fill_value": -1.0,
                    "dtype": var.dtype,
                    "level_out_name": self.level_out_name,
                    "level_out_size": field_level_out_size,
                    "time_size": self.time_size,
                    "num_cells": self.num_cells,
                }
                if field_level_out_size == 0:
                   if self.time_size == 0:
                      app = RaveField2d.model_validate(init_data)
                   else:
                      app = RaveField2d_plusTime.model_validate(init_data)
                else:
                   if self.time_size == 0:
                      app = RaveField3d.model_validate(init_data)
                   else:
                      app = RaveField3d_plusTime.model_validate(init_data)
                rave_fields.append(app)
        _LOGGER.debug(f"{rave_fields=}")
        return tuple(rave_fields)

    @staticmethod
    def _get_nc_attrs_(src: HasNcAttrsType) -> dict[str, Any]:
        # tdk: does valid_range matter?
        exclude = ("coordinates", "valid_range")
        return {
            ii: getattr(src, ii)
            for ii in src.ncattrs()
            if not ii.startswith("_") and ii not in exclude
        }


class FileDesc(BaseModel):
    path: Path
    origin: Literal["src", "dst"]
    field_names: tuple[str, ...]


class RaveToMpasRegridProcessor:
    _dst_mesh: esmpy.Mesh | None = None

    def __init__(self, context: RaveToMpasRegridContext) -> None:
        self.context = context

        self._regridder: esmpy.Regrid | None = None
        self._dst_field: FieldWrapper | None = None
        self._src_gwrap: GridWrapper | None = None

    def initialize(self) -> None:
        _LOGGER.info(f"initialize: {self.context=}")
        esmpy.Manager(debug=True)

        # if not self.context.scrip_path.exists() and self.context.rank == 0:
        #     _LOGGER.info("writing mpas scrip grid")
        #     from pyremap import MpasCellMeshDescriptor
        #
        #     mpas_desc = MpasCellMeshDescriptor(
        #         str(self.context.dst_path), self.context.mesh_name + ".init"
        #     )
        #     mpas_desc.to_scrip(str(self.context.scrip_path))

        pathsrc=self.context.src_path
        _LOGGER.info("create source grid")
        if self.context.x_corner_dim is None:
            self._src_gwrap = NcToGrid(
                path=pathsrc,
                spec=GridSpec(
                    x_center=self.context.x_center,
                    y_center=self.context.y_center,
                    x_dim=(self.context.x_dim,),
                    y_dim=(self.context.y_dim,),
                    x_corner=self.context.x_corner,
                    y_corner=self.context.y_corner,
                    x_corner_dim=self.context.x_corner_dim,
                    y_corner_dim=self.context.y_corner_dim,
                ),
            ).create_grid_wrapper()
        else:
            self._src_gwrap = NcToGrid(
                path=pathsrc,
                spec=GridSpec(
                    x_center=self.context.x_center,
                    y_center=self.context.y_center,
                    x_dim=(self.context.x_dim,),
                    y_dim=(self.context.y_dim,),
                    x_corner=self.context.x_corner,
                    y_corner=self.context.y_corner,
                    x_corner_dim=(self.context.x_corner_dim,),
                    y_corner_dim=(self.context.y_corner_dim,),
                ),
            ).create_grid_wrapper()

        _LOGGER.info("create source field")
        src_fwrap = self.create_src_field_wrapper(self.context.rave_fields[0].name)

        if self._dst_mesh is None:
            _LOGGER.info("create destination mesh")
            # dst_mesh = esmpy.Mesh(
            #     filename=str(self.context.scrip_path), filetype=esmpy.FileFormat.SCRIP
            # )
            self._dst_mesh = esmpy.Mesh(
                filename=str(self.context.scrip_path), filetype=esmpy.FileFormat.UGRID, meshname="grid_topology"
            )
        dst_mesh = self._dst_mesh
        local_bounds = reconcile_bounds((0, self._dst_mesh.size_owned[1]))

# Check for extra dims beyond lat/lon
        _LOGGER.info("create destination field")
        cells_dim = Dimension(name=("nCells",),
                          size=self.context.num_cells,
                          lower=local_bounds[0],
                          upper=local_bounds[1],
                          staggerloc=esmpy.MeshLoc.ELEMENT,
                          coordinate_type="element")
        dims = [cells_dim]
        time_dim = Dimension(name=("Time",), size=self.context.time_size, staggerloc=esmpy.StaggerLoc.CENTER, coordinate_type="time", lower=0, upper=self.context.time_size) if self.context.time_size > 0 else None
        level_dim = Dimension(name=(self.context.level_out_name,), size=self.context.level_out_size, staggerloc=esmpy.StaggerLoc.CENTER, coordinate_type="level", lower=0, upper=self.context.level_out_size) if self.context.level_out_size > 0 else None
        if self.context.level_out_size == 0:
        #2D
           if self.context.time_size == 0:
              # 2D, static in Time
              esmpy_dst_field = esmpy.Field(
                  dst_mesh, name="dst", meshloc=esmpy.MeshLoc.ELEMENT,
               )
           else:
              # 2D + Time
              esmpy_dst_field = esmpy.Field(
                  dst_mesh, name="dst", meshloc=esmpy.MeshLoc.ELEMENT, ndbounds=(self.context.time_size,)
               )
              dims.append(time_dim)
        else:
        #3D
           if self.context.time_size == 0:
              # 3D, static in Time
              esmpy_dst_field = esmpy.Field(
                  dst_mesh, name="dst", meshloc=esmpy.MeshLoc.ELEMENT, ndbounds=(self.context.level_out_size,)
              )
              dims.append(level_dim)
           else:
              # 3D + Time
              esmpy_dst_field = esmpy.Field(
                  dst_mesh, name="dst", meshloc=esmpy.MeshLoc.ELEMENT, ndbounds=(self.context.level_out_size, self.context.time_size)
              )
              dims.append(level_dim)
              dims.append(time_dim)
        gwrap = MeshWrapper(value=dst_mesh, dims=DimensionCollection(value=[cells_dim]))
        self._dst_field = FieldWrapper(value=esmpy_dst_field, gwrap=gwrap, dims=DimensionCollection(value=dims))

# Check for weights
        _LOGGER.info("create regridder")
        if self.context.weight_path.exists():
            _LOGGER.info("create regridder from file")
            self._regridder = esmpy.RegridFromFile(
                srcfield=src_fwrap.value,
                dstfield=self._dst_field.value,
                filename=str(self.context.weight_path),
            )
        else:
            _LOGGER.info("create regridder in-memory")
            if self.context.InterpMethod == "CONSERVE":
                _LOGGER.info("using 1st order conservative interp")
                self._regridder = esmpy.Regrid(
                    srcfield=src_fwrap.value,
                    dstfield=self._dst_field.value,
                    regrid_method=esmpy.RegridMethod.CONSERVE,
                    unmapped_action=esmpy.UnmappedAction.IGNORE,
                    ignore_degenerate=True,
                    large_file=True,
                    filename=str(self.context.weight_path),
                )
            elif self.context.InterpMethod == "CONSERVE_2ND":
                _LOGGER.info("using 2nd order conservative interp")
                self._regridder = esmpy.Regrid(
                    srcfield=src_fwrap.value,
                    dstfield=self._dst_field.value,
                    regrid_method=esmpy.RegridMethod.CONSERVE_2ND,
                    unmapped_action=esmpy.UnmappedAction.IGNORE,
                    ignore_degenerate=True,
                    large_file=True,
                    filename=str(self.context.weight_path),
                )
            elif self.context.InterpMethod == "BILINEAR":
                _LOGGER.info("using bilinear interp")
                self._regridder = esmpy.Regrid(
                    srcfield=src_fwrap.value,
                    dstfield=self._dst_field.value,
                    regrid_method=esmpy.RegridMethod.BILINEAR,
                    unmapped_action=esmpy.UnmappedAction.IGNORE,
                    ignore_degenerate=True,
                    large_file=True,
                    filename=str(self.context.weight_path),
                )
            else:
                _LOGGER.info("using nearest_STOD interp")
                self._regridder = esmpy.Regrid(
                    srcfield=src_fwrap.value,
                    dstfield=self._dst_field.value,
                    regrid_method=esmpy.RegridMethod.NEAREST_STOD,
                    unmapped_action=esmpy.UnmappedAction.IGNORE,
                    ignore_degenerate=True,
                    large_file=True,
                    filename=str(self.context.weight_path),
                )

    def run(self) -> None:
        _LOGGER.info("apply regridding")

        self.create_output_file()

        regridder = self.get_regridder()
        for rave_field in self.context.rave_fields:
            _LOGGER.info(f"regridding {rave_field.name=}")
            src_fwrap = self.create_src_field_wrapper(field_name=rave_field.name)

            dst_field = self.get_dst_field()
            # tdk: any more qa stuff? minimum threshold?
            dst_field.value.data.fill(0.0)
            regridder(src_fwrap.value, dst_field.value)
            # tdk: support NcToMesh
            dims = rave_field.create_dimension_collection(dst_field.gwrap.dims.value[0].bounds)
            _LOGGER.info(f"{dims=}")
            _LOGGER.info(f"writing field to netcdf")
            with open_nc(self.context.new_dst_path, mode="a") as ds:
                if self.context.dataset_name == "RAVE" and rave_field.name in ("FRP_MEAN", "FRE"):
                    area_dims = DimensionCollection(value=[dims.get("nCells")])
                    area_subset = load_variable_data(ds.variables['areaCell'], area_dims)
                    area_subset = area_subset.reshape(dst_field.dims.shape_local)
                    # area_subset = area_subset.reshape(dims.shape_local)
                    # area = np.asarray(ds.variables['areaCell'])
                    # area_subset = area[reconciled_bounds[0]:reconciled_bounds[1]].reshape(dims.shape_local)
                _LOGGER.info(f"creating variable {rave_field.name=}")
                var = ds.createVariable(
                    rave_field.name,
                    rave_field.dtype,
                    [dim.name[0] for dim in dims.value],
                    fill_value=rave_field.fill_value,
                )
                # Don't carry over fill value and datatype
                if self.context.dataset_name != 'GOES':
                    type_to_use = rave_field.dtype
                    for k, v in rave_field.attrs.items():
                       setattr(var, k, v)
                else:
                    type_to_use = np.float32

                _LOGGER.info(f"setting variable data {rave_field.name=}")
                # Multiply FRE/FRP by output area so it is back to W or J*s
                if self.context.dataset_name == "RAVE" and rave_field.name in ("FRP_MEAN", "FRE"):
                    set_variable_data(
                        var,
                        dst_field.dims,
                        dst_field.value.data * area_subset,
                        collective=True,
                    )
                elif self.context.dataset_name == "EROD":
                    # Explicitly check data dimensions to ensure shape alignment
                    erod_data = dst_field.value.data
                    _LOGGER.info(f"EROD Data incoming shape: {erod_data.shape}")

                    set_variable_data(
                        var,
                        dims,
                        dst_field.value.data,
                        collective=True,
                    )
                else:
                    set_variable_data(
                        var,
                        dst_field.dims,
                        dst_field.value.data,
                        collective=True,
                    )
            _LOGGER.info(f"finished writing field to netcdf {rave_field.name=}")
            src_fwrap.value.destroy()
            del src_fwrap

            if rave_field.name == "ENL_POLL":
                with open_nc(self.context.new_dst_path, mode="a") as ds:
                    _LOGGER.info(f"renaming and combining tree fields")

                    src_fwrap_enl = self.create_src_field_wrapper(field_name='ENL_POLL')
                    dst_field_enl = self.get_dst_field()
                    dst_field_enl.value.data.fill(0.0)
                    regridder(src_fwrap_enl.value, dst_field_enl.value)

                    src_fwrap_dbl = self.create_src_field_wrapper(field_name='DBL_POLL')
                    dst_field_dbl = self.get_dst_field()
                    dst_field_dbl.value.data.fill(0.0)
                    regridder(src_fwrap_dbl.value, dst_field_dbl.value)

                    rave_field = self.context.rave_fields[0]

                    var = ds.createVariable(
                        'TREE_POLL',
                        rave_field.dtype,
                        [dim.name[0] for dim in dims.value],
                        fill_value=rave_field.fill_value,
                    )
                    for k, v in self.context.rave_fields[0].attrs.items():
                        setattr(var, k, v)
                    set_variable_data(
                        var,
                        dst_field.dims,
                        dst_field_enl.value.data + dst_field_dbl.value.data,
                        collective=True,
                    )
                src_fwrap_enl.value.destroy()
                del src_fwrap_enl
                src_fwrap_dbl.value.destroy()
                del src_fwrap_dbl
            if rave_field.name == "TPM":
                with open_nc(self.context.new_dst_path, mode="a") as ds:
                    _LOGGER.info(f"calculating PM10 as TPM - PM25")
                    src_fwrap_ttl = self.create_src_field_wrapper(field_name='TPM')
                    src_fwrap_p25 = self.create_src_field_wrapper(field_name='PM25')

                    dst_field_ttl = self.get_dst_field()
                    dst_field_ttl.value.data.fill(0.0)
                    regridder(src_fwrap_ttl.value, dst_field_ttl.value)

                    dst_field_p25 = self.get_dst_field()
                    dst_field_p25.value.data.fill(0.0)
                    regridder(src_fwrap_p25.value, dst_field_p25.value)

                    rave_field = self.context.rave_fields[0]

                    var = ds.createVariable(
                        'PM10',
                        rave_field.dtype,
                        [dim.name[0] for dim in dims.value],
                        fill_value=rave_field.fill_value,
                    )
                    for k, v in self.context.rave_fields[0].attrs.items():
                        setattr(var, k, v)
                    data3 = dst_field_ttl.value.data - dst_field_p25.value.data
                    set_variable_data(
                        var,
                        dims,
                        data3,
                        collective=True,
                    )
                src_fwrap_ttl.value.destroy()
                del src_fwrap_ttl
                src_fwrap_p25.value.destroy()
                del src_fwrap_p25

        # if self.context.rank == 0:
        #     field_names = tuple(ii.name for ii in self.context.rave_fields)
        #     targets = [
        #         FileDesc(
        #             path=self.context.new_dst_path,
        #             origin="dst",
        #             field_names=field_names,
        #         ),
        #         FileDesc(
        #             path=self.context.src_path,
        #             origin="src",
        #             field_names=field_names,
        #         ),
        #     ]
        #     data_frame = self.create_desc_stuff(targets)
        #     data_frame.to_csv(self.context.desc_stats_out, index=False)

    def finalize(self) -> None:
        _LOGGER.info("finalizing")
        self._regridder.destroy()
        self._dst_field.value.destroy()
        self._src_gwrap.value.destroy()
        # self._dst_mesh.destroy()

    def create_desc_stuff(self, targets: Iterable[FileDesc]) -> pd.DataFrame:
        _LOGGER.info("entering create_desc_stuff")
        if self.context.rank > 0:
            raise ValueError

        to_concat = []
        for target in targets:
            with open_nc(target.path, mode="r", parallel=False) as ds:
                for varname in target.field_names:
                    data = ds.variables[varname][:].filled(np.nan).ravel()
                    data_frame = pd.DataFrame.from_dict({varname: data})
                    desc = data_frame.describe()
                    adds = {
                        varname: [
                            data_frame[varname].sum(),
                            data_frame[varname].isnull().sum(),
                            target.origin,
                            target.path,
                        ]
                    }
                    desc = pd.concat(
                        [
                            desc,
                            pd.DataFrame(
                                data=adds, index=["sum", "count_null", "origin", "path"]
                            ),
                        ]
                    )
                    to_concat.append(desc)
        ret = pd.concat([ii.transpose() for ii in to_concat])
        ret.index.name = "field_name"
        ret.reset_index(inplace=True)
        _LOGGER.info("exiting create_desc_stuff")
        return ret

    def create_src_field_wrapper(self, field_name: str) -> FieldWrapper:
        _LOGGER.info("create source field")
        if self.context.dataset_name == "GRA2PES" and field_name in ("h_agl",): # Special case for staggered grid
           src_fwrap = NcToField(
               path=self.context.src_path,
               name=field_name,
               gwrap=self.get_src_gwrap(),
               dim_time=(self.context.time_name,),
               dim_level=('bottom_top_stag',),
           ).create_field_wrapper()
        elif self.context.level_in_name == "None":
           if self.context.time_name == "None":
              src_fwrap = NcToField(
                  path=self.context.src_path,
                  name=field_name,
                  gwrap=self.get_src_gwrap(),
                  dim_time=None,
                  dim_level=None,
              ).create_field_wrapper()
           else:
              src_fwrap = NcToField(
                  path=self.context.src_path,
                  name=field_name,
                  gwrap=self.get_src_gwrap(),
                  dim_time=(self.context.time_name,),
                  dim_level=None,
              ).create_field_wrapper()
        else:
           if self.context.time_name == "None":
              src_fwrap = NcToField(
                  path=self.context.src_path,
                  name=field_name,
                  gwrap=self.get_src_gwrap(),
                  dim_time=None,
                  dim_level=(self.context.level_in_name,),
              ).create_field_wrapper()
           else:
              src_fwrap = NcToField(
                  path=self.context.src_path,
                  name=field_name,
                  gwrap=self.get_src_gwrap(),
                  dim_time=(self.context.time_name,),
                  dim_level=(self.context.level_in_name,),
              ).create_field_wrapper()

        # Get the area from the RAVE file, need to convert from /grid to /m2
        if (self.context.dataset_name == "RAVE" and field_name in ("PM25", "NH3", "SO2", "FRE", "FRP_MEAN", "TPM", "CH4", "CO", "NOx")):
            area_fwrap = NcToField(
                path=self.context.src_path,
                name='area',
                gwrap=self.get_src_gwrap(),
                dim_time=None,
            ).create_field_wrapper()
            area_data = area_fwrap.value.data

        # GRA2PES PM, convert from metric tons/km2/hr to ug/m2/s
        if self.context.dataset_name == "GRA2PES" and field_name in ("PM25-PRI", "PM10-PRI"):
            conv_aer = 1.e6 / 3600.
        # GRA2PES gases, convert from moles/km2/hr to moles/m2/s
        elif self.context.dataset_name == "GRA2PES" and field_name in ("HC01", "SO2", "CO", "NH3", "NOX"):
            conv_aer = 1.e-6 / 3600.
        # RAVE methane, convert from kg/hr to mol/m2/s
        elif self.context.dataset_name == "RAVE":
            if field_name == "CH4":
                conv_aer = (1.0 / 16.0) * 1000.
            elif field_name == "CO":
                conv_aer = (1.0 / 28.0) * 1000.
            elif field_name == "NH3":
                conv_aer = (1.0 / 17.0) * 1000.
            elif field_name == "NOx":
                conv_aer = ( (1.0 / 30.0) + (1.0 / 46.0) ) / 2. * 1000. 
            else:
                conv_aer = 1.0
        elif self.context.dataset_name == "NEMO_RWC" and field_name in ("PEC","POC","PMOTHR","PMC"):
            # Convert g/s/km2 (on 1km grid) to ug/m2/s -->
            conv_aer = 1.0
        elif self.context.dataset_name == "NEMO_ANTHRO" and field_name in ("PEC","POC","PMOTHR","PMC","PAL","PCA","PCL","PFE","PK","PMG","PMN","PNA","PNCOM","PNH4","PNO3","PSI","PSO4","PTI"):
            # Convert g/s/km2 to ug/m2/s -->
            conv_aer = 1.0
        elif self.context.dataset_name == "NEMO_ANTHRO" and field_name in ("CO","NO2","NO","HONO","CH4","NH3","NH3_FERT","TERP","ISOP","IVOC"):
            # Convert moles/km2/s to moles/m2/s -->
            conv_aer = 1.e-6
        else:
            conv_aer = 1.0

        src_data = src_fwrap.value.data
        if self.context.dataset_name == "RAVE" and field_name in ("PM25", "TPM"):
            # If RAVE aerosol emissions, convert from kg/hr to ug/m2/s
            src_data[:] = np.where(src_data < 0.0, 0.0, src_data * 1.e3 / area_data[:, :, np.newaxis] / 3600.)
        elif self.context.dataset_name == "RAVE" and field_name in ("CH4", "NH3", "SO2", "CO", "NOx"):
            # If RAVE gas emissions, convert from kg/hr to mol/m2/s
            src_data[:] = np.where(src_data < 0.0, 0.0, conv_aer * src_data / area_data[:, :, np.newaxis] / 3600.)
        elif self.context.dataset_name == "RAVE" and field_name in ("FRE", "FRP_MEAN"):
            # For FRE, FRP, don't multiply area by 1.e6, cancelled out by MW to W conversion
            src_data[:] = np.where(src_data < 0.0, 0.0, src_data / (area_data[:, :, np.newaxis]))
        else:
            src_data[:] = np.where(src_data < 0.0, 0.0, conv_aer * src_data)

        src_data[:] = np.where(np.isnan(src_data), 0.0, src_data)
        return src_fwrap

    def get_src_gwrap(self) -> GridWrapper:
        if self._src_gwrap is None:
            raise ValueError
        return self._src_gwrap

    def get_dst_field(self) -> FieldWrapper:
        if self._dst_field is None:
            raise ValueError
        return self._dst_field

    def get_regridder(self) -> esmpy.Regrid:
        if self._regridder is None:
            raise ValueError
        return self._regridder

    def create_output_file(self) -> None:
        _LOGGER.info("create output file")
        if self.context.rank != 0:
            return

        with open_nc(self.context.new_dst_path, mode="w", clobber=True, parallel=False) as dst_nc:
            dst_nc.createDimension("nCells", self.context.num_cells)
            if self.context.level_out_name != "None":
                dst_nc.createDimension(self.context.level_out_name, self.context.level_out_size)
            dst_nc.createDimension("StrLen", 64)
            if self.context.time_size > 1:
                dst_nc.createDimension("Time", self.context.time_size)
            elif self.context.time_size == 1:
                if "Time" not in dst_nc.dimensions:
                    dst_nc.createDimension("Time")
                _LOGGER.info("Not creating a time dimension")
            dst_nc.setncattr("created_at", str(datetime.now(timezone.utc)))
            dst_nc.setncattr("src_path", str(self.context.src_path))
            dst_nc.setncattr("dst_path", str(self.context.dst_path))

            if self.context.dataset_name in ("RAVE", "NGFS"):
                base_vars = ("latCell", "lonCell", "areaCell", "xtime")
            elif self.context.dataset_name in ("FENGSHA_2D", "EROD"):
                base_vars = ("latCell", "lonCell")
            else:
                base_vars = ("latCell", "lonCell", "xtime")

            with open_nc(self.context.dst_path, mode="r", parallel=False) as src_nc:
                for varname in base_vars:
                    if varname in src_nc.variables:
                        copy_nc_variable(src_nc, dst_nc, varname, copy_data=True)
                    else:
                        _LOGGER.warning(
                            f"Destination static file does not contain {varname}; skipping copy."
                        )

    def init_destination_only(self) -> None:
        """Loads the heavy MPAS destination mesh once for dynamic NGFS processing."""
        if self._dst_field is not None:
            return

        _LOGGER.info("Initializing MPAS Destination Mesh (Once)")
        esmpy.Manager(debug=True)

        #if not self.context.scrip_path.exists() and self.context.rank == 0:
        #     _LOGGER.info("writing mpas scrip grid")
        #     mpas_desc = MpasCellMeshDescriptor(
        #         str(self.context.dst_path), self.context.mesh_name + ".init"
        #     )
        #     mpas_desc.to_scrip(str(self.context.scrip_path))

        if self._dst_mesh is None:
            _LOGGER.info("create destination mesh")
            self._dst_mesh = esmpy.Mesh(
                filename=str(self.context.scrip_path), filetype=esmpy.FileFormat.UGRID,
                meshname="grid_topology"
            )
        dst_mesh = self._dst_mesh
        local_bounds = reconcile_bounds((0, dst_mesh.size_owned[1]))

        _LOGGER.info("create destination field")
        cells_dim = Dimension(name=("nCells",),
                          size=self.context.num_cells,
                          lower=local_bounds[0],
                          upper=local_bounds[1],
                          staggerloc=esmpy.MeshLoc.ELEMENT,
                          coordinate_type="element")
        dims = [cells_dim]
        time_dim = Dimension(name=("Time",), size=self.context.time_size, staggerloc=esmpy.StaggerLoc.CENTER, coordinate_type="time", lower=0, upper=self.context.time_size) if self.context.time_size > 0 else None
        level_dim = Dimension(name=(self.context.level_out_name,), size=self.context.level_out_size, staggerloc=esmpy.StaggerLoc.CENTER, coordinate_type="level", lower=0, upper=self.context.level_out_size) if self.context.level_out_size > 0 else None
        if self.context.level_out_size == 0:
            if self.context.time_size == 0:
                esmpy_dst_field = esmpy.Field(dst_mesh, name="dst", meshloc=esmpy.MeshLoc.ELEMENT)
            else:
                esmpy_dst_field = esmpy.Field(
                    dst_mesh, name="dst", meshloc=esmpy.MeshLoc.ELEMENT, ndbounds=(self.context.time_size,)
                )
                dims.append(time_dim)
        else:
            if self.context.time_size == 0:
                esmpy_dst_field = esmpy.Field(
                    dst_mesh, name="dst", meshloc=esmpy.MeshLoc.ELEMENT, ndbounds=(self.context.level_out_size,)
                )
                dims.append(level_dim)
            else:
                esmpy_dst_field = esmpy.Field(
                    dst_mesh, name="dst", meshloc=esmpy.MeshLoc.ELEMENT, ndbounds=(self.context.level_out_size, self.context.time_size)
                )
                dims.append(level_dim)
                dims.append(time_dim)
        gwrap = MeshWrapper(value=dst_mesh, dims=DimensionCollection(value=[cells_dim]))
        self._dst_field = FieldWrapper(value=esmpy_dst_field, gwrap=gwrap, dims=DimensionCollection(value=dims))

    def process_ngfs_file(self, file_path: Path, resolution: float = 0.01) -> None:
        """Dynamically builds a mesh for NGFS points, regrids, and writes the output."""
        _LOGGER.info(f"Processing NGFS file: {file_path}")
        self.init_destination_only()

        # 1. Read NGFS Coordinates AND Area
        with open_nc(file_path, mode="r") as ds:
            lats = ds.variables['lat'][:].filled(np.nan)
            lons = ds.variables['lon'][:].filled(np.nan)

            # Read the NGFS area (in km2)
            if 'GRID_AREA' in ds.variables:
                grid_area = ds.variables['GRID_AREA'][:].filled(np.nan)
            elif "PIXEL_AREA" in ds.variables:
                grid_area = ds.variables["PIXEL_AREA"][:].filled(np.nan)
            else:
                _LOGGER.warning("GRID_AREA not found! Defaulting to 1.0 km2.")
                grid_area = np.ones_like(lats)

        # Filter out NaNs
        valid = ~np.isnan(lats) & ~np.isnan(lons) & ~np.isnan(grid_area)
        lats = lats[valid]
        lons = lons[valid]
        grid_area = grid_area[valid]

        # CRITICAL FIX: Convert -180/180 to 0/360 to match MPAS grid
        lons = lons % 360.0

        if len(lats) == 0:
            _LOGGER.warning("No valid fires in file.")
            return

        # 2. Build Sparse Source Mesh
        src_mesh = create_ngfs_sparse_mesh(lats, lons, resolution)
        if src_mesh is None:
            return

        # 3. Create Output NetCDF File using the same header rules as RAVE.
        self.create_output_file()
        dst_field = self.get_dst_field()

        # 4. Process Each Variable
        for rave_field in self.context.rave_fields:
            _LOGGER.info(f"regridding NGFS {rave_field.name=}")

            # Create Source Field dynamically
            src_field = esmpy.Field(src_mesh, name=rave_field.name, meshloc=esmpy.MeshLoc.ELEMENT)

            # Map MPAS expected name to NGFS actual name
            if rave_field.name == "PM25":
                ngfs_var_name = "EMIS_PM25"
            else:
                ngfs_var_name = rave_field.name

            # Load the raw data
            with open_nc(file_path, mode="r") as ds:
                if ngfs_var_name in ds.variables:
                    raw_data = ds.variables[ngfs_var_name][:].filled(0.0)[valid]
                else:
                    _LOGGER.warning(f"Variable {ngfs_var_name} not found! Skipping.")
                    continue

            # ---------------------------------------------------------
            # UNIT CONVERSIONS (Identical to RAVE logic)
            # ---------------------------------------------------------
            if rave_field.name in ("PM25", "TPM"):
                # Convert from kg/hr to ug/m2/s (1e3 handles the km2 to m2 and kg to ug ratio)
                src_data = np.where(raw_data < 0.0, 0.0, raw_data * 1.e3 / grid_area / 3600.0)
            elif rave_field.name in ("FRE", "FRP_MEAN"):
                # For FRE, FRP: MW to W (1e6) cancels out with km2 to m2 (1e6)
                src_data = np.where(raw_data < 0.0, 0.0, raw_data / grid_area)
            else:
                src_data = np.where(raw_data < 0.0, 0.0, raw_data)

            src_field.data[:] = src_data

            # Create Dynamic Regridder
            regridder = esmpy.Regrid(
                srcfield=src_field,
                dstfield=dst_field.value,
                regrid_method=esmpy.RegridMethod.CONSERVE,
                unmapped_action=esmpy.UnmappedAction.IGNORE
            )

            # Apply Regridding
            dst_field.value.data.fill(0.0)
            regridder(src_field, dst_field.value)

            # Write to Output NetCDF
            reconciled_bounds = dst_field.dims.value[0].bounds
            dims = rave_field.create_dimension_collection(reconciled_bounds)

            with open_nc(self.context.new_dst_path, mode="a") as ds:
                output_var_name = NGFS_OUTPUT_NAMES.get(rave_field.name, rave_field.name)
                var = ds.createVariable(
                    output_var_name,
                    rave_field.dtype,
                    [dim.name[0] for dim in dims.value],
                    fill_value=rave_field.fill_value,
                )
                for k, v in rave_field.attrs.items():
                    setattr(var, k, v)

                # Multiply by areaCell for Power/Energy variables (back to total W in cell)
                if rave_field.name in ("FRP_MEAN", "FRE"):
                    area_dims = DimensionCollection(value=[dims.get("nCells")])
                    area_subset = load_variable_data(ds.variables['areaCell'], area_dims)
                    area_subset = area_subset.reshape(dst_field.dims.shape_local)
                    set_variable_data(var, dims, rave_field.reshape_field_data(dst_field.value.data * area_subset), collective=True)
                else:
                    set_variable_data(var, dims, rave_field.reshape_field_data(dst_field.value.data), collective=True)

            # Clean up memory
            regridder.destroy()
            src_field.destroy()

        # Clean up mesh
        src_mesh.destroy()


def main(ctx: ChemRegridContext) -> None:
    dataset_name = ctx.dataset_name.value  # Which dataset are we interpolating?
    workdir = ctx.workdir  # Directory where operations will be processed
    input_dir = ctx.input_dir  # Top directory of input data
    output_dir = ctx.output_dir  # Top directory of output data
    cycle = ctx.cycle
    scrip_path = ctx.rw_scrip_path # Cycle Time, YYYYMMDDHH
    dst_path = ctx.rw_dst_path
    mesh_name = ctx.mesh_name
    ebb_dcycle = ctx.ebb_dcycle

    desc_stats_out = ctx.rw_desc_stats_out
    #
    YYYY = cycle[0:4]
    MM = cycle[4:6]
    DD = cycle[6:8]
    HH = cycle[8:10]
    x = datetime(int(YYYY), int(MM), int(DD), int(HH), 0, 0)
    JJJ = x.strftime("%j")
    DOWh = int(x.strftime("%u"))
    if DOWh <= 5:
        DOWs = "weekdy"
    elif DOWh == 6:
        DOWs = "satdy"
    else:
        DOWs = "sundy"

    # Calculate the number of cells in the
    with open_nc(dst_path, mode="r", parallel=False) as src_nc:
        foo = src_nc.variables['latCell']
        num_cells = len(foo)
        # xland = src_nc.variables['xland']
        # lmask[:] = np.where(xland > 0,1,0)

    if dataset_name == "RAVE":
        field_names = ("TPM", "FRE", "FRP_MEAN", "PM25", "NH3", "SO2", "CH4","CO","NOx")
        # JLS, TODO - NEED TO ACCOUNT FOR EBB1, MORE THAN 24, ETC.
        # Determine the cycle dates to process +%Y%m%d%H
        dates_needed = []
        for i in range(25):
            if ebb_dcycle == 1: # Same-day emissions
                x = datetime(int(YYYY), int(MM), int(DD), int(HH), 0, 0) + timedelta(hours=i)
            elif ebb_dcycle == -1 or ebb_dcycle == 2: # Persistence (-1) or forecasted (2) needs prev 24 hours 
                x = datetime(int(YYYY), int(MM), int(DD), int(HH), 0, 0) - timedelta(hours=i)
            else:
                _LOGGER.info("EBB_DCYLE selection not recognized, reverting to same day, ebb_dcycle = 1")
                x = datetime(int(YYYY), int(MM), int(DD), int(HH), 0, 0) + timedelta(hours=i)
                
            y = x.strftime("%Y%m%d%H")
            dates_needed.append(y)
        #
        x_center = "grid_lont"
        y_center = "grid_latt"
        x_dim = "grid_xt"
        y_dim = "grid_yt"
        x_corner = "grid_lon"
        y_corner = "grid_lat"
        x_corner_dim = "grid_x"
        y_corner_dim = "grid_y"
        level_in_name = "None"
        # level_in_size = None
        level_out_name = "nkwildfire"
        level_out_size = 1
        time_name = "time"
        time_size = 1
        InterpMethod = "CONSERVE"
    elif dataset_name == "NGFS":
        field_names = ("FRE", "FRP_MEAN", "PM25")

        # Determine the cycle dates to process +%Y%m%d%H
        # This is for RETROS (using current datetime, not day before)
        dates_needed = []
        for i in range(25): # GAF retro current day emissions
            if ebb_dcycle == 1: # Same-day emissions
                x = datetime(int(YYYY), int(MM), int(DD), int(HH), 0, 0) + timedelta(hours=i)
            elif ebb_dcycle == -1 or ebb_dcycle == 2: # Persistence (-1) or forecasted (2) needs prev 24 hours
                x = datetime(int(YYYY), int(MM), int(DD), int(HH), 0, 0) - timedelta(hours=i)
            else:
                _LOGGER.info("EBB_DCYLE selection not recognized, reverting to same day, ebb_dcycle = 1")
                x = datetime(int(YYYY), int(MM), int(DD), int(HH), 0, 0) + timedelta(hours=i)
            y = x.strftime("%Y%m%d%H")
            dates_needed.append(y)
        #
        x_center = "lon"
        y_center = "lat"
        x_dim = "point" # Dummy dimension name for context
        y_dim = "point" # Dummy dimension name for context
        # We set corners to None because the helper calculates them in memory
        x_corner = None
        y_corner = None
        x_corner_dim = None
        y_corner_dim = None
        level_in_name = "None"
        level_out_name = "nkwildfire"
        level_out_size = 1
        time_name = "time"
        time_size = 1
        InterpMethod = "CONSERVE"
    elif dataset_name == "GRA2PES":
        field_names = ("PM25-PRI", "PM10-PRI","SO2","CO","NOX","NH3","h_agl")  # ,"HC01"=methane BAQMS, summer, 2025
        x_center = "XLONG"  # "XLONG_M"
        y_center = "XLAT"  # "XLAT_M"
        x_dim = "west_east"
        y_dim = "south_north"
        x_corner = "XLONG_C"
        y_corner = "XLAT_C"
        x_corner_dim = "west_east_stag"
        y_corner_dim = "south_north_stag"
        level_in_name = "bottom_top"
        level_out_name = "nkanthro"
        level_out_size = 20
        time_name = "Time"
        time_size = 12
        InterpMethod = "CONSERVE"
        # InterpMethod = "BILINEAR"
    elif dataset_name == "NEMO_ANTHRO":
        field_names = ("POC", "PEC", "PMOTHR", "PMC","CO","NH3","SO2","NO","NO2","PAL","PCA","PCL","PFE","PK","PMG","PMN","PNA","PNCOM","PNH4","PNO3","PSI","PSO4","PTI")
        x_center = "lon"
        y_center = "lat"
        x_dim = "COL"
        y_dim = "ROW"
        x_corner = "lonc"
        y_corner = "latc"
        x_corner_dim = "COLC"
        y_corner_dim = "ROWC"
        level_in_name = "LAY"
        level_out_name = "nkanthro"
        level_out_size = 1
        time_name = "TSTEP"
        time_size = 1
        InterpMethod = "CONSERVE"
#       InterpMethod = "BILINEAR"
    elif dataset_name == "NEMO_RWC":
        field_names = ("POC", "PEC", "PMOTHR", "PMC")
        x_center = "lon"
        y_center = "lat"
        x_dim = "COL"
        y_dim = "ROW"
        x_corner = "lonc"
        y_corner = "latc"
        x_corner_dim = "COLC"
        y_corner_dim = "ROWC"
        level_in_name = "None"
        level_out_name = "None"
        level_out_size = 0
        time_name = "Time"
        time_size = 1
        InterpMethod = "CONSERVE"
#       InterpMethod = "BILINEAR"
    elif dataset_name == "PECM":
        field_names = ("DBL_POLL", "ENL_POLL", "GRA_POLL", "RAG_POLL")
        x_center = "lon"
        y_center = "lat"
        x_dim = "COL"
        y_dim = "ROW"
        x_corner = "lonc"
        y_corner = "latc"
        x_corner_dim = "COLC"
        y_corner_dim = "ROWC"
        level_in_name = "None"
        level_out_name = "nkbiogenic"
        level_out_size = 1
        time_name = "time"
        time_size = 1
        InterpMethod = "CONSERVE"
    elif dataset_name == "ECOREGION":
        field_names = ("ecoregion_ID",)
        x_center = "geolon"
        y_center = "geolat"
        x_dim = "lon"
        y_dim = "lat"
        x_corner = None
        y_corner = None
        x_corner_dim = None
        y_corner_dim = None
        level_in_name = "None"
        level_out_name = "nkwildfire"
        level_out_size = 1
        time_name = "time"
        time_size = 1
        InterpMethod = "NEAREST_STOD"
    elif dataset_name == "NARR":
        field_names = ("RWC_denominator",)
        x_center = "lon"
        y_center = "lat"
        x_dim = "x"
        y_dim = "y"
        x_corner = None
        y_corner = None
        x_corner_dim = None
        y_corner_dim = None
        level_in_name = "None"
        level_out_name = "None"
        level_out_size = 0
        time_name = "Time"
        time_size = 1
        InterpMethod = "BILINEAR"
    elif dataset_name == "FENGSHA_2D":
        field_names = ("clayfrac", "sandfrac", "uthres", "ssm")
        x_center = "longitude"
        y_center = "latitude"
        x_dim = "lon"
        y_dim = "lat"
        x_corner = None
        y_corner = None
        x_corner_dim = None
        y_corner_dim = None
        level_in_name = "None"
        level_out_name = "None"
        level_out_size = 0
        time_name = "None"
        time_size = 0
        InterpMethod = "BILINEAR"
    elif dataset_name == "FENGSHA_2D_Time":
        field_names = ("rdrag",)
        x_center = "longitude"
        y_center = "latitude"
        x_dim = "lon"
        y_dim = "lat"
        x_corner = None
        y_corner = None
        x_corner_dim = None
        y_corner_dim = None
        level_in_name = "None"
        level_out_name = "None"
        level_out_size = 0
        time_name = "time"
        time_size = 12
        InterpMethod = "BILINEAR"
    elif dataset_name == "EROD":
        field_names = ("erod",)
        x_center = "longitude"
        y_center = "latitude"
        x_dim = "lon"
        y_dim = "lat"
        x_corner = None
        y_corner = None
        x_corner_dim = None
        y_corner_dim = None
        level_in_name = "nSoilTypes"
        level_out_name = "nSoilTypes"
        level_out_size = 3
        time_name = "None"
        time_size = 0
        InterpMethod = "BILINEAR"
    elif dataset_name == "FMC":  # fuel moisture content
        field_names = ("10h_dead_fuel_moisture_content",)
        dates_needed = []
        for i in range(25):
            x = datetime(int(YYYY), int(MM), int(DD), int(HH), 0, 0) - timedelta(hours=i)
            y = x.strftime("%Y%m%d%H")
            dates_needed.append(y)
        x_center = "longitude"
        y_center = "latitude"
        x_dim = "nx"
        y_dim = "ny"
        x_corner = None
        y_corner = None
        x_corner_dim = None
        y_corner_dim = None
        level_in_name = "None"
        level_out_name = "nkwildfire"
        level_out_size = 1
        time_name = "time"
        time_size = 1
        InterpMethod = "BILINEAR"
    elif "GOES" in dataset_name:
        if "AOD" in dataset_name:
           field_names = ("AOD",)
           prod_code = "AOD"
        elif "ADP" in dataset_name:
           field_names = ("Smoke",)
           prod_code = "ADP"
        else:
           raise Exception("Other GOES products are not yet supported. You'll need to add them.")
        x_center = "longitude"
        y_center = "latitude"
        x_dim = "columns"
        y_dim = "rows"
        x_corner = None
        y_corner = None
        x_corner_dim = None
        y_corner_dim = None
        level_in_name = "None"
        level_out_name = "None"
        level_out_size = 0
        time_name = "None"
        time_size = 0
        InterpMethod = "BILINEAR"
        if "WEST" in dataset_name:
           sat_num = "18"
           coord_file_name = "goes18_abi_conus_lat_lon.nc"
        elif "EAST" in dataset_name:
           sat_num = "19"
           coord_file_name = "goes19_abi_conus_interpolated_lat_lon.nc"
        else:
           raise Exception("Dataset name does not specify a satellite location. Must include either 'EAST' or 'WEST'")
        dates_needed = []
        for i in range(25):
            if ebb_dcycle == 1: # Same-day emissions
                x = datetime(int(YYYY), int(MM), int(DD), int(HH), 0, 0) + timedelta(hours=i)
            elif ebb_dcycle == -1 or ebb_dcycle == 2: # Persistence (-1) or forecasted (2) needs prev 24 hours 
                x = datetime(int(YYYY), int(MM), int(DD), int(HH), 0, 0) - timedelta(hours=i)
            else:
                _LOGGER.info("EBB_DCYLE selection not recognized, reverting to same day, ebb_dcycle = 1")
                x = datetime(int(YYYY), int(MM), int(DD), int(HH), 0, 0) - timedelta(hours=i)
            y = x.strftime("%Y%m%d%H")
            dates_needed.append(y)

    weight_path = ctx.get_weight_path(InterpMethod)

    if dataset_name == "RAVE":
        processor = None
        for date_to_process in dates_needed:
            _LOGGER.info(f"RAVE processing {date_to_process=}")
            rave_paths = find_latest_rave_file(input_dir, date_to_process, ebb_dcycle, dataset_name, max_lookback_hours=24)
            if not rave_paths:
                _LOGGER.warn(
                    f"No matching files found for {date_to_process} (even after lookback).")
                continue

            _LOGGER.info(f'Reading RAVE file: {rave_paths=}')
            rave_path = rave_paths[0]
            new_dst_path = output_dir / (mesh_name + "-RAVE-" + date_to_process + ".nc")

            # --- OPTIMIZATION START ---
            if processor is None:
                _LOGGER.info("FIRST PASS: Full Initialization")
                # This pays the "expensive" cost of loading weights/grids, but only once.

                context = RaveToMpasRegridContext(
                    dataset_name=dataset_name,
                    workdir=workdir,
                    src_path=rave_path,
                    dst_path=dst_path,
                    new_dst_path=new_dst_path,
                    desc_stats_out=desc_stats_out,
                    weight_path=weight_path,
                    InterpMethod=InterpMethod,
                    scrip_path=scrip_path,
                    num_cells=num_cells,
                    mesh_name=mesh_name,
                    field_names=field_names,
                    x_center=x_center,
                    y_center=y_center,
                    x_dim=x_dim,
                    y_dim=y_dim,
                    x_corner=x_corner,
                    y_corner=y_corner,
                    x_corner_dim=x_corner_dim,
                    y_corner_dim=y_corner_dim,
                    level_in_name=level_in_name,
                    # level_in_size=level_in_size,
                    level_out_name=level_out_name,
                    level_out_size=level_out_size,
                    time_name=time_name,
                    time_size=time_size
    
                )
                processor = RaveToMpasRegridProcessor(context=context)
                processor.initialize()
            else:
                _LOGGER.info("SUBSEQUENT PASSES: Hot Swap")
                # Just update the paths in the existing context.
                # The grids and regridder (weights) remain loaded in memory.
                processor.context.src_path = rave_path
                processor.context.new_dst_path = new_dst_path
            # Run the regridding (Fast)
            processor.run()
            # --- OPTIMIZATION END ---
            # Only finalize after ALL files are done
        if processor:
            processor.finalize()

            _LOGGER.info("success")

    elif dataset_name == "NGFS":
        # Initialize context with dummy paths (they get overwritten in the loop)
        context = RaveToMpasRegridContext(
            dataset_name=dataset_name,
            workdir=workdir,
            src_path=Path("dummy"),
            dst_path=dst_path,
            new_dst_path=Path("dummy"),
            desc_stats_out=desc_stats_out,
            weight_path=weight_path,
            InterpMethod=InterpMethod,
            scrip_path=scrip_path,
            num_cells=num_cells,
            mesh_name=mesh_name,
            field_names=field_names,
            x_center=x_center, y_center=y_center, x_dim=x_dim, y_dim=y_dim,
            x_corner=x_corner, y_corner=y_corner,
            x_corner_dim=x_corner_dim, y_corner_dim=y_corner_dim,
            level_in_name=level_in_name, level_out_name=level_out_name, level_out_size=level_out_size,
            time_name=time_name, time_size=time_size
        )

        processor = RaveToMpasRegridProcessor(context=context)

        for date_to_process in dates_needed:
            
            ngfs_paths = glob.glob(str(input_dir) + "/ngfs_processed_" + date_to_process + "0000.nc")

            if not ngfs_paths:
                print(f"ERROR: Missing NGFS file for {date_to_process}. Skipping.")
                exit(1)
                # TODO: perhaps add a helper similarly as I added for RAVE to search for the latest
                # available file in case that the current datetime does not exist
                continue

            ngfs_path = Path(ngfs_paths[0])
            new_dst_path = Path(str(output_dir) + "/" + mesh_name + "-NGFS-" + date_to_process + ".nc")
            print(f"GAF reading NGFS file: {ngfs_path}")

            # Update context paths for the current hour
            processor.context.src_path = ngfs_path
            processor.context.new_dst_path = new_dst_path

            # Execute the dynamic regridding for this specific hour's fires
            # Note that resolution is hard coded...
            processor.process_ngfs_file(ngfs_path, resolution=0.01)

        _LOGGER.info("NGFS success")

    elif "GOES" in dataset_name:
        processor = None
        date_to_process = dates_needed[0]
        rave_paths = find_latest_rave_file(input_dir, date_to_process, -1, dataset_name, max_lookback_hours=2, sat_num=sat_num, product=prod_code)
        files_to_cat = rave_paths
        _LOGGER.info(f"will cat files: {files_to_cat=}")
        if COMM.rank == 0:
           with xr.open_mfdataset(files_to_cat, combine='nested', concat_dim='file') as ds:
               # 2. Calculate the nanmean across the new 'file' dimension
               # skipna=True (default) ensures it behaves like np.nanmean
               ds_averaged = ds[field_names[0]].mean(dim='file', skipna=True)
           # _LOGGER.debug(ds_averaged)
           ds_averaged.encoding.update({
              'dtype': 'float32',
              '_FillValue': -999
           })
           ds_averaged.drop_vars(['x','y'])
           ds_averaged = ds_averaged.rename({'y':'rows','x':'columns'})
           ds_averaged.to_netcdf(output_dir / f'GOES_{prod_code}_merged.nc')
           coord_file = f"{workdir}/{coord_file_name}"
           if not isfile(coord_file):
              raise Exception(
 """If processing GOES AOD/ADP data, you must provide the GOES coordinate data files, which
 are not provided by this repository. Instructions for obtaining them can be found in README.GOES in the root directory.""")
           ds_coords = xr.open_dataset(coord_file,engine='netcdf4')
           ds_coords.to_netcdf(output_dir / f'GOES_{prod_code}_merged.nc',mode='a')

        if not rave_paths:
            msg = f"No matching GOES files found for {date_to_process} (even after lookback)."
            _LOGGER.error(msg)
            raise ValueError(msg)

        _LOGGER.info(f'Reading merged GOES file: GOES_{prod_code}_merged.nc')
        #rave_path = rave_paths[0]
        rave_path = output_dir / f"GOES_{prod_code}_merged.nc"
        new_dst_path = output_dir / (mesh_name + "-GOES-" + date_to_process + ".nc")
        # --- OPTIMIZATION START ---
        if processor is None:
            # FIRST PASS: Full Initialization
            # This pays the "expensive" cost of loading weights/grids, but only once.

            context = RaveToMpasRegridContext(
                dataset_name=dataset_name,
                workdir=workdir,
                src_path=rave_path,
                dst_path=dst_path,
                new_dst_path=new_dst_path,
                desc_stats_out=desc_stats_out,
                weight_path=weight_path,
                InterpMethod=InterpMethod,
                scrip_path=scrip_path,
                num_cells=num_cells,
                mesh_name=mesh_name,
                field_names=field_names,
                x_center=x_center,
                y_center=y_center,
                x_dim=x_dim,
                y_dim=y_dim,
                x_corner=x_corner,
                y_corner=y_corner,
                x_corner_dim=x_corner_dim,
                y_corner_dim=y_corner_dim,
                level_in_name=level_in_name,
                level_out_name=level_out_name,
                level_out_size=level_out_size,
                time_name=time_name,
                time_size=time_size

            )
            processor = RaveToMpasRegridProcessor(context=context)
            processor.initialize()
        else:
            # SUBSEQUENT PASSES: Hot Swap
            # Just update the paths in the existing context.
            # The grids and regridder (weights) remain loaded in memory.
            processor.context.src_path = rave_path
            processor.context.new_dst_path = new_dst_path
        # Run the regridding (Fast)
        processor.run()
        # --- OPTIMIZATION END ---
                # Only finalize after ALL files are done
        if processor:
            processor.finalize()

        _LOGGER.info("success")

    elif dataset_name == "FMC":
        for date_to_process in dates_needed:
            rave_paths = glob.glob(str(input_dir / ("fmc_" + date_to_process + ".nc")))
            rave_path = Path(rave_paths[0])
            new_dst_path = output_dir / ("fmc_" + date_to_process + "_" + mesh_name + ".nc")

            context = RaveToMpasRegridContext(
                dataset_name=dataset_name,
                workdir=workdir,
                src_path=rave_path,
                dst_path=dst_path,
                new_dst_path=new_dst_path,
                desc_stats_out=desc_stats_out,
                weight_path=weight_path,
                InterpMethod=InterpMethod,
                scrip_path=scrip_path,
                num_cells=num_cells,
                mesh_name=mesh_name,
                field_names=field_names,
                x_center=x_center,
                y_center=y_center,
                x_dim=x_dim,
                y_dim=y_dim,
                x_corner=x_corner,
                y_corner=y_corner,
                x_corner_dim=x_corner_dim,
                y_corner_dim=y_corner_dim,
                level_in_name=level_in_name,
                level_out_name=level_out_name,
                level_out_size=level_out_size,
                time_name=time_name,
                time_size=time_size

            )
            processor = RaveToMpasRegridProcessor(context=context)
            processor.initialize()
            processor.run()
            processor.finalize()

            _LOGGER.info("success")
#
    elif dataset_name == "GRA2PES":
        rave_path = input_dir / ("GRA2PESv1.0_total_2021" + MM + "_" + DOWs + "_00to11Z.nc")
        new_dst_path = output_dir / (dataset_name + "v1.0_total_" + mesh_name + "_00to11Z.nc")
        context = RaveToMpasRegridContext(
            dataset_name=dataset_name,
            workdir=workdir,
            src_path=rave_path,
            dst_path=dst_path,
            new_dst_path=new_dst_path,
            desc_stats_out=desc_stats_out,
            weight_path=weight_path,
            InterpMethod=InterpMethod,
            scrip_path=scrip_path,
            num_cells=num_cells,
            mesh_name=mesh_name,
            field_names=field_names,
            x_center=x_center,
            y_center=y_center,
            x_dim=x_dim,
            y_dim=y_dim,
            x_corner=x_corner,
            y_corner=y_corner,
            x_corner_dim=x_corner_dim,
            y_corner_dim=y_corner_dim,
            level_in_name=level_in_name,
            level_out_name=level_out_name,
            level_out_size=level_out_size,
            time_name=time_name,
            time_size=time_size

        )
        processor = RaveToMpasRegridProcessor(context=context)
        processor.initialize()
        processor.run()
        processor.finalize()

        _LOGGER.info("success")

        rave_path = input_dir / ("GRA2PESv1.0_total_2021" + MM + "_" + DOWs + "_12to23Z.nc")
        new_dst_path = output_dir / (dataset_name + "v1.0_total_" + mesh_name + "_12to23Z.nc")
        context = RaveToMpasRegridContext(
            dataset_name=dataset_name,
            workdir=workdir,
            src_path=rave_path,
            dst_path=dst_path,
            new_dst_path=new_dst_path,
            desc_stats_out=desc_stats_out,
            weight_path=weight_path,
            InterpMethod=InterpMethod,
            scrip_path=scrip_path,
            num_cells=num_cells,
            mesh_name=mesh_name,
            field_names=field_names,
            x_center=x_center,
            y_center=y_center,
            x_dim=x_dim,
            y_dim=y_dim,
            x_corner=x_corner,
            y_corner=y_corner,
            x_corner_dim=x_corner_dim,
            y_corner_dim=y_corner_dim,
            level_in_name=level_in_name,
            level_out_name=level_out_name,
            level_out_size=level_out_size,
            time_name=time_name,
            time_size=time_size

        )
        processor = RaveToMpasRegridProcessor(context=context)
        processor.initialize()
        processor.run()
        processor.finalize()

        _LOGGER.info("success")

    else:
        if dataset_name == "PECM":
            rave_path = input_dir / ("pollen_obs_" + YYYY + "_BELD6_ef_T_" + JJJ + ".nc")
            new_dst_path = output_dir / ("pollen_ef_" + mesh_name + "_" + YYYY + "_" + JJJ + ".nc")
        elif dataset_name == "NEMO_RWC":
            rave_path = input_dir / "NEMO_RWC_POC_PEC_PMOTHR.annual.2017.nc"
            new_dst_path = output_dir / ("NEMO_RWC_ANNUAL_TOTAL_" + mesh_name + ".nc")
        elif dataset_name == "NEMO_ANTHRO":
            rave_path = input_dir / ("NEMO_ANTHRO_" + mesh_name + "_" + YYYY + MM + DD + HH + "_SECTORSUM.nc")
            new_dst_path = output_dir / ("NEMO_ANTHRO_" + mesh_name + ".nc")
        elif dataset_name == "NARR":
            rave_path = input_dir / "rwc_emission_denominator.2017.nc"
            new_dst_path = output_dir / ("NEMO_RWC_DENOMINATOR_2017_" + mesh_name + ".nc")
        elif dataset_name == "ECOREGION":
            rave_path = input_dir / "veg_map.nc"
            new_dst_path = output_dir / ("ecoregions_" + mesh_name + "_mpas.nc")
        elif dataset_name == "FENGSHA_2D":
            rave_path = input_dir / "FENGSHA_RRFS_NA_3km_2026_2D.nc"
            new_dst_path = output_dir / ("fengsha_dust_inputs.2D."+ mesh_name + ".nc")
        elif dataset_name == "FENGSHA_2D_Time":
            rave_path = input_dir / "FENGSHA_RRFS_NA_3km_2026_2D_Time.nc"
            new_dst_path = output_dir / ("fengsha_dust_inputs.2D_Time."+ mesh_name + ".nc")
        elif dataset_name == "EROD":
            rave_path = input_dir / "EROD_RRFS_NA_3km_2026.nc"
            new_dst_path = output_dir / ("dust_inputs.erod."+ mesh_name + ".nc")

        context = RaveToMpasRegridContext(
            dataset_name=dataset_name,
            workdir=workdir,
            src_path=rave_path,
            dst_path=dst_path,
            new_dst_path=new_dst_path,
            desc_stats_out=desc_stats_out,
            weight_path=weight_path,
            InterpMethod=InterpMethod,
            scrip_path=scrip_path,
            num_cells=num_cells,
            mesh_name=mesh_name,
            field_names=field_names,
            x_center=x_center,
            y_center=y_center,
            x_dim=x_dim,
            y_dim=y_dim,
            x_corner=x_corner,
            y_corner=y_corner,
            x_corner_dim=x_corner_dim,
            y_corner_dim=y_corner_dim,
            level_in_name=level_in_name,
            # level_in_size=level_in_size,
            level_out_name=level_out_name,
            level_out_size=level_out_size,
            time_name=time_name,
            time_size=time_size

        )
        processor = RaveToMpasRegridProcessor(context=context)
        processor.initialize()
        processor.run()
        processor.finalize()

        _LOGGER.info("success")
