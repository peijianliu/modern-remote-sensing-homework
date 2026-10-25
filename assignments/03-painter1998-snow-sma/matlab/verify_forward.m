function verify_forward(output_path, prepared_workspace)
% 以 4 半径 x 2 角度 x 2 冰选项校核显式输入，输出 16 条数值记录。
% prepared_workspace 应包含官方光学库和 prepare_optics.py 生成的工作副本。
assert(~isfile(output_path), 'Output exists; select a new verification file');
case_root = fileparts(fileparts(mfilename('fullpath')));
baseline = load(fullfile(case_root, 'data/demo/forward_models.mat'));
config = jsondecode(fileread(fullfile(case_root, 'configs/baseline.json')));
args = pure_snow_inputs(config);
fields = fieldnames(args);
assert(isequal(sort(fields), sort(fieldnames(baseline.input_args))));
variable_fields = {'rds_snw', 'ice_ri', 'coszen'};
for i = 1:numel(fields)
    name = fields{i};
    if ~ismember(name, variable_fields)
        assert(isequal(args.(name), baseline.input_args.(name)), ['Default mismatch: ' name]);
    end
end
previous = pwd;
cleanup = onCleanup(@() cd(previous));
cd(prepared_workspace);
addpath(fullfile(prepared_workspace, 'data/cache/matlab'));
records = [];
for ice = [1 2]
    for radius = [50 200 500 1000]
        for angle = [0 60]
            args.ice_ri = ice;
            args.rds_snw = radius;
            args.coszen = cosd(double(angle));
            result = snicarAD_v3(args);
            ri = find(baseline.radii_um == radius);
            ai = find(baseline.angles_deg == angle);
            oi = find(baseline.ice_options == ice);
            reference = reshape(baseline.snicar_albedo(ri,:,ai,oi), [], 1);
            difference = max(abs(result.albedo(:) - reference));
            broadband_difference = abs(result.alb_slr - baseline.broadband_snicar(ri,ai,oi));
            records = [records; ice radius angle difference broadband_difference]; %#ok<AGROW>
        end
    end
end
assert(size(records,1) == 16 && max(records(:,4:5), [], 'all') < 1e-10);
summary = struct('input_field_count', numel(fields), 'cases', size(records,1), ...
    'columns', {{'ice_option','radius_um','angle_deg','spectral_max_abs','broadband_abs'}}, ...
    'results', records, 'maximum_absolute_error', max(records(:,4:5), [], 'all'));
fid = fopen(output_path, 'w', 'n', 'UTF-8');
assert(fid >= 0);
file_cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', jsonencode(summary, PrettyPrint=true));
fprintf('SNICAR explicit input validation: %d cases, max error %.3g\n', size(records,1), summary.maximum_absolute_error);
end
