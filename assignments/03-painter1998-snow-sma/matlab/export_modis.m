function export_modis(input_file, output_file)
% 读取 MOD09GA HDF-EOS 的二维网格字段。
if nargin < 1
    input_file = 'data/raw/MOD09GA.A2013123.h08v05.061.2021229143331.hdf';
end
if nargin < 2, output_file = 'data/cache/modis_fields.mat'; end
assert(~isfile(output_file), 'Existing output preserved: use a new version');
information = hdfinfo(input_file, 'eos');
grid500 = information.Grid(strcmp({information.Grid.Name}, 'MODIS_Grid_500m_2D'));
grid1000 = information.Grid(strcmp({information.Grid.Name}, 'MODIS_Grid_1km_2D'));
band_raw = zeros(2400, 2400, 7, 'int16');
field_names = cell(1,7);
for band = 1:7
    prefix = sprintf('sur_refl_b%02d', band);
    name = find_field(grid500, prefix);
    field_names{band} = name;
    band_raw(:,:,band) = read_field(input_file, grid500.Name, name);
end
qc_name = find_field(grid500, 'QC_500m');
qc = uint32(read_field(input_file, grid500.Name, qc_name));
state = uint16(read_field(input_file, grid1000.Name, find_field(grid1000, 'state_1km')));
solar_zenith_raw = int16(read_field(input_file, grid1000.Name, find_field(grid1000, 'SolarZenith')));
sensor_zenith_raw = int16(read_field(input_file, grid1000.Name, find_field(grid1000, 'SensorZenith')));
orbit_pointer = read_field(input_file, grid1000.Name, find_field(grid1000, 'orbit_pnt'));
granule_pointer = read_field(input_file, grid1000.Name, find_field(grid1000, 'granule_pnt'));
assert(isequal(size(state), [1200 1200]) && isequal(size(qc), [2400 2400]));
% MATLAB hdfread exposes grid dimensions in listed YDim, XDim order.
dimension_order = {grid500.DataFields(2).Dims.Name};
assert(strcmp(dimension_order{1}, 'YDim') && strcmp(dimension_order{2}, 'XDim'));
save(output_file, 'band_raw', 'qc', 'state', 'solar_zenith_raw', ...
     'sensor_zenith_raw', 'orbit_pointer', 'granule_pointer', 'dimension_order', ...
     'field_names', 'qc_name', '-v7');
% Confirm attributes from the actual SDS rather than relying only on the guide.
sid = hdf('SD','start',input_file,'read');
did = hdf('SD','select',sid,hdf('SD','nametoindex',sid,qc_name));
[qc_fill,status] = hdf('SD','readattr',did,hdf('SD','findattr',did,'_FillValue'));
assert(status == 0);
hdf('SD','endaccess',did);
did = hdf('SD','select',sid,hdf('SD','nametoindex',sid,field_names{1}));
[refl_fill,s1] = hdf('SD','readattr',did,hdf('SD','findattr',did,'_FillValue'));
[scale,s2] = hdf('SD','readattr',did,hdf('SD','findattr',did,'scale_factor'));
[valid_range,s3] = hdf('SD','readattr',did,hdf('SD','findattr',did,'valid_range'));
assert(all([s1 s2 s3] == 0));
hdf('SD','endaccess',did); hdf('SD','end',sid);
attributes_path = fullfile(fileparts(output_file),'modis_attributes.mat');
if ~isfile(attributes_path)
    save(attributes_path,'qc_fill','refl_fill','scale','valid_range','-v7');
end
fprintf('Exported HDF-EOS: %s; orientation YDim x XDim\n', output_file);
end

function name = find_field(grid, prefix)
names = {grid.DataFields.Name};
found = names(startsWith(names, [prefix '_']));
assert(numel(found) == 1, 'Ambiguous/missing field: %s', prefix);
name = found{1};
end

function value = read_field(file, grid, field)
value = hdfread(file, grid, 'Fields', field);
if iscell(value), value = value{1}; end
end
