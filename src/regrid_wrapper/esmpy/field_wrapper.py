import abc
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Literal, Sequence, Tuple, Union

import esmpy
import netCDF4 as nc
import numpy as np
from mpi4py import MPI

from regrid_wrapper.context.comm import COMM, Tag, reconcile_bounds
from regrid_wrapper.context.logging import LOGGER

_LOGGER = LOGGER.getChild(__name__)


@contextmanager
def open_nc(path: Path, mode: Literal["r", "w", "a"] = "r", clobber: bool = False, parallel: bool = True) -> Iterator[nc.Dataset]:
    _LOGGER.debug(f"opening {path}")
    kwds = dict(
        mode=mode,
        clobber=clobber,
        parallel=parallel,
        comm=MPI.COMM_WORLD,
        format="NETCDF4_CLASSIC",
        info=MPI.Info(),
    )
    _LOGGER.debug(f"{kwds=}")
    ds = nc.Dataset(path, **kwds)  # type: ignore[arg-type]
    try:
        yield ds
    finally:
        ds.close()


HasNcAttrsType = Union[nc.Dataset, nc.Variable]


def copy_nc_attrs(src: HasNcAttrsType, dst: HasNcAttrsType) -> None:
    for attr in src.ncattrs():
        if attr.startswith("_"):
            continue
        setattr(dst, attr, getattr(src, attr))


def resize_nc(
    src_path: Path,
    dst_path: Path,
    new_sizes: Dict[str, int],
    copy_values_for: Sequence[str] = tuple(),
) -> None:
    with open_nc(src_path, mode="r") as src:
        with open_nc(dst_path, mode="w") as dst:
            copy_nc_attrs(src, dst)
            for dim in src.dimensions:
                size = get_aliased_key(new_sizes, dim)
                dst.createDimension(dim, size=size)
            for varname in src.variables.keys():
                copy_data = varname in copy_values_for
                copy_nc_variable(src, dst, varname, copy_data=copy_data)


def copy_nc_variable(src: nc.Dataset, dst: nc.Dataset, varname: str, copy_data: bool = False) -> None:
    var = src.variables[varname]
    fill_value = getattr(var, "_FillValue") if hasattr(var, "_FillValue") else None
    new_var = dst.createVariable(varname, var.dtype, var.dimensions, fill_value=fill_value)
    copy_nc_attrs(var, new_var)
    if copy_data:
        new_var[:] = var[:]


NameListType = Tuple[str, ...]


def get_aliased_key(source: Dict, keys: NameListType | str) -> Any:
    if isinstance(keys, str):
        keys_to_find: NameListType = (keys,)
    else:
        keys_to_find = keys
    for key in keys_to_find:
        try:
            return source[key]
        except KeyError:
            continue
    raise ValueError(f"key not found: {keys}")


def get_nc_dimension(ds: nc.Dataset, names: NameListType) -> nc.Dimension:
    return get_aliased_key(ds.dimensions, names)


@dataclass
class Dimension:
    name: NameListType
    size: int
    lower: int
    upper: int
    staggerloc: int
    coordinate_type: Literal["y", "x", "time", "element", "level"]

    @property
    def bounds(self) -> tuple[int, int]:
        return self.lower, self.upper


@dataclass
class DimensionCollection:
    value: Tuple[Dimension, ...]

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(dim.size for dim in self.value)

    @property
    def shape_local(self) -> tuple[int, ...]:
        return tuple(dim.upper - dim.lower for dim in self.value)

    def get(self, name: str | NameListType) -> Dimension:
        if isinstance(name, str):
            name_to_find: NameListType = (name,)
        else:
            name_to_find = name
        for jj in name_to_find:
            for ii in self.value:
                if jj in ii.name:
                    return ii
        raise ValueError(f"dimension not found: {name}")


def create_dimension_map(dims: DimensionCollection) -> Dict[str, int]:
    ret = {}
    for idx, dim in enumerate(dims.value):
        for name in dim.name:
            ret[name] = idx
    return ret


def load_variable_data(
    var: nc.Variable,
    target_dims: DimensionCollection,
) -> np.ndarray:
    slices = []
    for ii in var.dimensions:
        try:
            dim_bound = target_dims.get(ii)
            slices.append(slice(dim_bound.lower, dim_bound.upper))
        except ValueError:
            slices.append(0) if ii == "nSoilTypes" else slices.append(slice(None))
    raw_data = var[*slices]
    dim_map = {dim: ii for ii, dim in enumerate(var.dimensions)}
    axes = [get_aliased_key(dim_map, ii.name) for ii in target_dims.value]
    transposed_data = raw_data.transpose(axes)
    return transposed_data


def set_variable_data(
    var: nc.Variable,
    target_dims: DimensionCollection,
    target_data: np.ndarray,
    collective: bool = False,
) -> np.ndarray:
    if collective:
        var.set_collective(True)
    dim_map = create_dimension_map(target_dims)
    try:
        axes = [get_aliased_key(dim_map, ii) for ii in var.dimensions]
    except:
        _LOGGER.debug(f"{dim_map=}; {var.dimensions=}")
        raise
    transposed_data = target_data.transpose(axes)
    slices = [slice(target_dims.get(ii).lower, target_dims.get(ii).upper) for ii in var.dimensions]
    _LOGGER.debug(f"var.shape: {var.shape}")
    _LOGGER.debug(f"transposed_data.shape: {transposed_data.shape}")
    _LOGGER.debug(f"slices: {slices}")
    var[*slices] = transposed_data
    if collective:
        var.set_collective(False)
    return transposed_data


def set_variable_data_serial(path: Path, varname: str, target_dims: DimensionCollection, target_data: np.ndarray) -> None:
    if COMM.rank > 0:
        COMM.recv(source=COMM.rank - 1, tag=Tag.SET_VARIABLE_DATA)
    with open_nc(path, mode="a", parallel=False) as ds:
        var = ds.variables[varname]
        set_variable_data(var, target_dims, target_data)
    if COMM.rank < COMM.size - 1:
        COMM.send(True, dest=COMM.rank + 1, tag=Tag.SET_VARIABLE_DATA)
    COMM.barrier()


@dataclass
class AbstractWrapper(abc.ABC):
    dims: DimensionCollection


@dataclass
class GridSpec:
    x_center: str
    y_center: str
    x_dim: NameListType
    y_dim: NameListType
    x_corner: str | None = None
    y_corner: str | None = None
    x_corner_dim: NameListType | None = None
    y_corner_dim: NameListType | None = None
    x_index: int = 0
    y_index: int = 1

    def __post_init__(self) -> None:
        corner_meta = [
            self.x_corner,
            self.y_corner,
            self.x_corner_dim,
            self.y_corner_dim,
        ]
        is_given_sum = sum([ii is not None for ii in corner_meta])
        if is_given_sum > 0 and is_given_sum != len(corner_meta):
            raise ValueError("if one corner name is supplied, then all must be supplied")

    @property
    def has_corners(self) -> bool:
        return self.x_corner is not None

    def get_x_corner(self) -> str:
        if self.x_corner is None:
            raise ValueError
        return self.x_corner

    def get_y_corner(self) -> str:
        if self.y_corner is None:
            raise ValueError
        return self.y_corner

    def get_x_data(self, grid: esmpy.Grid, staggerloc: esmpy.StaggerLoc) -> np.ndarray:
        return grid.get_coords(self.x_index, staggerloc=staggerloc)

    def get_y_data(self, grid: esmpy.Grid, staggerloc: esmpy.StaggerLoc) -> np.ndarray:
        return grid.get_coords(self.y_index, staggerloc=staggerloc)

    def create_grid_dims(self, ds: nc.Dataset, grid: esmpy.Grid, staggerloc: esmpy.StaggerLoc) -> DimensionCollection:
        if staggerloc == esmpy.StaggerLoc.CENTER:
            x_dim, y_dim = self.x_dim, self.y_dim
        elif staggerloc == esmpy.StaggerLoc.CORNER:
            if self.x_corner_dim is None or self.y_corner_dim is None:
                raise ValueError
            x_dim, y_dim = self.x_corner_dim, self.y_corner_dim
        else:
            raise NotImplementedError(staggerloc)
        x_dimobj = Dimension(
            name=x_dim,
            size=get_nc_dimension(ds, x_dim).size,
            lower=grid.lower_bounds[staggerloc][self.x_index],
            upper=grid.upper_bounds[staggerloc][self.x_index],
            staggerloc=staggerloc,
            coordinate_type="x",
        )
        y_dimobj = Dimension(
            name=y_dim,
            size=get_nc_dimension(ds, y_dim).size,
            lower=grid.lower_bounds[staggerloc][self.y_index],
            upper=grid.upper_bounds[staggerloc][self.y_index],
            staggerloc=staggerloc,
            coordinate_type="y",
        )
        if self.x_index == 0:
            value = [x_dimobj, y_dimobj]
        elif self.x_index == 1:
            value = [y_dimobj, x_dimobj]
        else:
            raise NotImplementedError(self.x_index, self.y_index)
        return DimensionCollection(value=tuple(value))


@dataclass
class GridWrapper(AbstractWrapper):
    value: esmpy.Grid
    spec: GridSpec
    corner_dims: DimensionCollection | None = None

    def fill_nc_variables(self, path: Path) -> None:
        if self.corner_dims is not None:
            raise NotImplementedError
        staggerloc = esmpy.StaggerLoc.CENTER
        with open_nc(path, "a") as ds:
            x_center_data = self.spec.get_x_data(self.value, staggerloc)
            set_variable_data(ds.variables[self.spec.x_center], self.dims, x_center_data)
            y_center_data = self.spec.get_y_data(self.value, staggerloc)
            set_variable_data(ds.variables[self.spec.y_center], self.dims, y_center_data)


@dataclass
class MeshWrapper(AbstractWrapper):
    value: esmpy.Mesh


@dataclass
class NcToMesh:
    path: Path
    filetype: int = esmpy.FileFormat.UGRID
    meshname: str = "grid_topology"

    def create_mesh_wrapper(self) -> MeshWrapper:
        t1 = time.perf_counter()
        mesh = esmpy.Mesh(
            filename=str(self.path),
            filetype=self.filetype,
            meshname=self.meshname,
        )
        t2 = time.perf_counter()
        LOGGER.debug(f"mesh read time: {t2 - t1} s, {COMM.size=}")
        local_bounds = reconcile_bounds((0, mesh.size_owned[1]))
        LOGGER.debug(f"{local_bounds=}")

        with open_nc(self.path, "r") as ds:
            host = ds.variables[self.meshname]
            n_faces = ds.dimensions[host.face_dimension].size
            dim = Dimension(
                name=(host.face_dimension,),
                size=n_faces,
                lower=local_bounds[0],
                upper=local_bounds[1],
                staggerloc=esmpy.MeshLoc.ELEMENT,
                coordinate_type="element",
            )
            dims = DimensionCollection(value=(dim,))

        mwrap = MeshWrapper(value=mesh, dims=dims)
        return mwrap

    def __post_init__(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(self.path)


@dataclass
class NcToGrid:
    path: Path
    spec: GridSpec

    def create_grid_wrapper(self) -> GridWrapper:
        with open_nc(self.path, "r") as ds:
            grid_shape = self._create_grid_shape_(ds)
            staggerloc = esmpy.StaggerLoc.CENTER
            grid = esmpy.Grid(
                grid_shape,
                staggerloc=staggerloc,
                coord_sys=esmpy.CoordSys.SPH_DEG,
            )
            dims = self.spec.create_grid_dims(ds, grid, staggerloc)
            grid_x_center_coords = self.spec.get_x_data(grid, staggerloc)
            grid_x_center_coords[:] = load_variable_data(ds.variables[self.spec.x_center], dims)
            grid_y_center_coords = self.spec.get_y_data(grid, staggerloc)
            grid_y_center_coords[:] = load_variable_data(ds.variables[self.spec.y_center], dims)

            if self.spec.has_corners:
                corner_dims = self._add_corner_coords_(ds, grid)
            else:
                corner_dims = None

            gwrap = GridWrapper(value=grid, dims=dims, spec=self.spec, corner_dims=corner_dims)
            return gwrap

    def _create_grid_shape_(self, ds: nc.Dataset) -> np.ndarray:
        x_size = get_nc_dimension(ds, self.spec.x_dim).size
        y_size = get_nc_dimension(ds, self.spec.y_dim).size
        if self.spec.x_index == 0:
            grid_shape = (x_size, y_size)
        elif self.spec.x_index == 1:
            grid_shape = (y_size, x_size)
        else:
            raise NotImplementedError(self.spec.x_index, self.spec.y_index)
        return np.array(grid_shape)

    def _add_corner_coords_(self, ds: nc.Dataset, grid: esmpy.Grid) -> DimensionCollection:
        staggerloc = esmpy.StaggerLoc.CORNER
        grid.add_coords(staggerloc)
        dims = self.spec.create_grid_dims(ds, grid, staggerloc)
        grid_x_corner_coords = self.spec.get_x_data(grid, staggerloc)
        if self.spec.x_corner is None:
            raise ValueError
        grid_x_corner_coords[:] = load_variable_data(ds.variables[self.spec.x_corner], dims)
        grid_y_corner_coords = self.spec.get_y_data(grid, staggerloc)
        if self.spec.y_corner is None:
            raise ValueError
        grid_y_corner_coords[:] = load_variable_data(ds.variables[self.spec.y_corner], dims)
        return dims


GeomType = GridWrapper | MeshWrapper


@dataclass
class FieldWrapper(AbstractWrapper):
    value: esmpy.Field
    gwrap: GeomType

    def fill_nc_variable(self, path: Path) -> None:
        _LOGGER.debug(r"filling variable: {self.value.name}")
        with open_nc(path, "a") as ds:
            var = ds.variables[self.value.name]
            set_variable_data(var, self.dims, self.value.data)


@dataclass
class MetaToField:
    name: str
    gwrap: GeomType
    staggerloc: int = esmpy.StaggerLoc.CENTER
    dim_time: Dimension | None = None
    dim_level: Dimension | None = None

    def create_field_wrapper(self) -> FieldWrapper:
        if self.dim_time is None and self.dim_level is not None:
            raise ValueError("dim_time must be given if dim_level is given")
        ndbounds = None
        dims = list(self.gwrap.dims.value)
        ndbounds_dims = []
        if self.dim_time is not None:
            ndbounds_dims = [self.dim_time]
            ndbounds = [self.dim_time.size]
            if self.dim_level is not None:
                ndbounds_dims = [self.dim_level, self.dim_time]
                ndbounds.append(self.dim_level.size)
        field = esmpy.Field(
            self.gwrap.value,
            name=self.name,
            ndbounds=ndbounds,
            staggerloc=self.staggerloc,
        )
        target_dims = DimensionCollection(value=tuple(dims + ndbounds_dims))
        return FieldWrapper(value=field, dims=target_dims, gwrap=self.gwrap)


@dataclass
class NcToField:
    path: Path
    name: str
    gwrap: GeomType
    dim_time: NameListType | None = None
    dim_level: NameListType | None = None
    staggerloc: int = esmpy.StaggerLoc.CENTER
    load_field_data_from_file: bool = True

    def create_field_wrapper(self) -> FieldWrapper:
        with open_nc(self.path, "r") as ds:
            if self.dim_time is None:
                if self.dim_level is None:
                    ndbounds: tuple[int, ...] | None = None
                    target_dims = self.gwrap.dims
                else:
                    ndbounds = (len(get_nc_dimension(ds, self.dim_level)),)
                    level_dim = Dimension(
                        name=self.dim_level,
                        size=ndbounds[0],
                        lower=0,
                        upper=ndbounds[0],
                        staggerloc=self.staggerloc,
                        coordinate_type="level",
                    )
                    target_dims = DimensionCollection(value=tuple(list(self.gwrap.dims.value) + [level_dim]))
            else:
                if self.dim_level is None:
                    ndbounds = (len(get_nc_dimension(ds, self.dim_time)),)
                    time_dim = Dimension(
                        name=self.dim_time,
                        size=ndbounds[0],
                        lower=0,
                        upper=ndbounds[0],
                        staggerloc=self.staggerloc,
                        coordinate_type="time",
                    )
                    target_dims = DimensionCollection(value=tuple(list(self.gwrap.dims.value) + [time_dim]))
                else:
                    ndbounds_base = (len(get_nc_dimension(ds, self.dim_time)),)
                    time_dim = Dimension(
                        name=self.dim_time,
                        size=ndbounds_base[0],
                        lower=0,
                        upper=ndbounds_base[0],
                        staggerloc=self.staggerloc,
                        coordinate_type="time",
                    )
                    ndbounds_base = (len(get_nc_dimension(ds, self.dim_level)),)
                    level_dim = Dimension(
                        name=self.dim_level,
                        size=ndbounds_base[0],
                        lower=0,
                        upper=ndbounds_base[0],
                        staggerloc=self.staggerloc,
                        coordinate_type="level",
                    )
                    target_dims = DimensionCollection(value=tuple(list(self.gwrap.dims.value) + [level_dim] + [time_dim]))
                    ndbounds = (
                        len(get_nc_dimension(ds, self.dim_level)),
                        len(get_nc_dimension(ds, self.dim_time)),
                    )
            if isinstance(self.gwrap, GridWrapper):
                field = esmpy.Field(
                    self.gwrap.value,
                    name=self.name,
                    ndbounds=ndbounds,
                    staggerloc=self.staggerloc,
                )
            elif isinstance(self.gwrap, MeshWrapper):
                field = esmpy.Field(
                    self.gwrap.value,
                    name=self.name,
                    ndbounds=ndbounds,
                    meshloc=self.staggerloc,
                )
            else:
                raise NotImplementedError(type(self.gwrap))
            if self.load_field_data_from_file:
                field.data[:] = load_variable_data(ds.variables[self.name], target_dims)
            fwrap = FieldWrapper(value=field, dims=target_dims, gwrap=self.gwrap)
            return fwrap


@dataclass
class FieldWrapperCollection:
    value: Tuple[FieldWrapper, ...]

    def fill_nc_variables(self, path: Path) -> None:
        for fwrap in self.value:
            fwrap.fill_nc_variable(path)

    def __post_init__(self) -> None:
        if len(set([id(ii.value.grid) for ii in self.value])) != 1:
            raise ValueError("all fields must share the same grid")
