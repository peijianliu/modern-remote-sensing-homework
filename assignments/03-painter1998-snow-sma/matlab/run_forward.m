function run_forward(output_path, angles_deg)
% 在相同光学输入下计算 SNICAR-ADv3 与 WW80 半无限雪层光谱。
if nargin < 1, output_path = 'data/cache/forward_models.mat'; end
if nargin < 2, angles_deg = [0 60 22]; end
assert(~isfile(output_path), 'Output exists; select a new version to preserve it');
addpath('data/cache/matlab');
addpath('vendor/SPIRES');
config = jsondecode(fileread('configs/baseline.json'));
driver = fileread('vendor/SNICARv3/snicarAD_v3_drv.m');
driver = extractBefore(driver, '% CALL SNICAR WITH INPUT ARGUMENTS');
driver = regexprep(driver, '(?m)^clear;\s*', '');
eval(driver);
input_args.sno_shp = 1;
input_args.R_sfc_all_wvl = 0;
input_args.dz = config.forward.depth_m;
input_args.rho_snw = config.forward.density_kg_m3;
input_args.atm = config.forward.atmosphere;
radii_um = 30:10:1500;
wavelength_um = 0.205:0.01:4.995;
ice_options = [1 2];
shape = [numel(radii_um), numel(wavelength_um), numel(angles_deg), 2];
snicar_albedo = zeros(shape);
ww_albedo = zeros(shape);
broadband_snicar = zeros(numel(radii_um), numel(angles_deg), 2);
broadband_ww = zeros(size(broadband_snicar));
solar_flux = zeros(numel(angles_deg), numel(wavelength_um));
depth_differences = [];
paths = jsondecode(fileread('data/sources/optics_extraction.json'));
for optical_index = 1:2
    input_args.ice_ri = ice_options(optical_index);
    if optical_index == 1, ice_stub = 'ice_Wrn84'; else, ice_stub = 'ice_Wrn08'; end
    for radius_index = 1:numel(radii_um)
        input_args.rds_snw = radii_um(radius_index);
        ice_file = sprintf('%s%s/%s_%04d.nc', paths.optics_root, ice_stub, ice_stub, input_args.rds_snw);
        omega = double(ncread(ice_file, 'ss_alb'));
        asymmetry = double(ncread(ice_file, 'asm_prm'));
        for angle_index = 1:numel(angles_deg)
            input_args.coszen = cosd(angles_deg(angle_index));
            di = snicarAD_v3(input_args);
            spectrum = twostream(input_args.coszen, omega, asymmetry, 'delta-Eddington');
            snicar_albedo(radius_index, :, angle_index, optical_index) = di.albedo(:);
            ww_albedo(radius_index, :, angle_index, optical_index) = spectrum(:);
            broadband_snicar(radius_index, angle_index, optical_index) = di.alb_slr;
            flux = di.flx_dwn_spc(:);
            solar_flux(angle_index, :) = flux;
            broadband_ww(radius_index, angle_index, optical_index) = sum(flux .* spectrum(:)) / sum(flux);
            if ismember(input_args.rds_snw, [50 200 500 1000]) && angle_index <= 2
                input_args.dz = 200;
                doubled = snicarAD_v3(input_args);
                difference = max(abs(doubled.albedo(:) - di.albedo(:)));
                depth_differences(end+1,:) = [optical_index input_args.rds_snw angles_deg(angle_index) difference]; %#ok<AGROW>
                input_args.dz = config.forward.depth_m;
            end
        end
        if mod(radius_index, 20) == 0
            fprintf('forward: ice=%d radius=%g um\n', optical_index, input_args.rds_snw);
        end
    end
end
assert(all(isfinite(snicar_albedo(:))) && all(isfinite(ww_albedo(:))));
assert(min(snicar_albedo(:)) >= -1e-8 && max(snicar_albedo(:)) <= 1+1e-8);
assert(max(depth_differences(:,4)) < 1e-4, '100m layer not optically thick');
save(output_path, 'radii_um', 'wavelength_um', 'angles_deg', 'ice_options', ...
    'snicar_albedo', 'ww_albedo', 'broadband_snicar', 'broadband_ww', ...
    'solar_flux', 'depth_differences', 'input_args', '-v7');
fprintf('Saved forward spectra: %s; max depth difference=%g\n', output_path, max(depth_differences(:,4)));
end
