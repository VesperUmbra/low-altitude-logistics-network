function extract_followup_logs_to_csv()
% Extract trajectory-relevant fields from 2025-2026 FCC flight logs
% and export unified point-level and flight-level CSV tables.

workspace_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
data_root = fullfile(workspace_root, '.tmp', 'new_flight_logs');
review_data_dir = fullfile(workspace_root, 'for_review', 'review_data');
if ~exist(review_data_dir, 'dir')
    mkdir(review_data_dir);
end

points_csv = fullfile(review_data_dir, 'followup_trajectory_points_2025_2026.csv');
summary_csv = fullfile(review_data_dir, 'followup_flight_summary_2025_2026.csv');

grid.min_lon = 113.3109;
grid.max_lon = 114.2486;
grid.min_lat = 22.3138;
grid.max_lat = 22.8700;
grid.grid_size = 100.0;
grid.lat_center = (grid.min_lat + grid.max_lat) / 2;
grid.meters_per_degree_lat = 111000.0;
grid.meters_per_degree_lon = 111000.0 * cosd(grid.lat_center);
lat_range_m = (grid.max_lat - grid.min_lat) * grid.meters_per_degree_lat;
lon_range_m = (grid.max_lon - grid.min_lon) * grid.meters_per_degree_lon;
grid.n_rows = ceil(lat_range_m / grid.grid_size);
grid.n_cols = ceil(lon_range_m / grid.grid_size);

mat_files = collect_mat_files(data_root);
if isempty(mat_files)
    error('No .mat files found under %s', data_root);
end

% keep only 2025-2026 logs
keep = false(numel(mat_files), 1);
for i = 1:numel(mat_files)
    [~, stem, ~] = fileparts(mat_files{i});
    keep(i) = length(stem) >= 8 && ...
        (strncmp(stem, '2025', 4) || strncmp(stem, '2026', 4));
end
mat_files = mat_files(keep);
mat_files = sort(mat_files);

tmp_csv = fullfile(tempdir, 'followup_chunk.csv');
first_chunk = true;
summary_rows = {};
summary_header = { ...
    'source_folder', 'route_code', 'file_name', 'flight_id', ...
    'flight_date_local', 'flight_start_local', 'flight_end_local', ...
    'duration_s', 'n_valid_points', 'n_waypoints', ...
    'start_lon', 'start_lat', 'end_lon', 'end_lat', ...
    'start_alt_agl', 'end_alt_agl', 'median_ground_speed' ...
};

fprintf('Extracting %d follow-up logs...\n', numel(mat_files));
for i = 1:numel(mat_files)
    file_path = mat_files{i};
    [source_folder, route_code, file_name, flight_id, flight_start_local] = parse_file_metadata(data_root, file_path);
    payload = load(file_path, 'FCC');
    if ~isfield(payload, 'FCC') || ~isfield(payload.FCC, 'log_navi')
        fprintf('Skipping %s (missing FCC.log_navi)\n', file_name);
        continue;
    end

    navi = payload.FCC.log_navi;
    lat = column_or_nan(navi, 'latitude');
    lon = column_or_nan(navi, 'longitude');
    alt_msl = column_or_nan(navi, 'alt_msl');
    alt_agl = column_or_nan(navi, 'altitude_agl');
    vel_x = column_or_nan(navi, 'VelX');
    vel_y = column_or_nan(navi, 'VelY');
    vel_z = column_or_nan(navi, 'VelZ');
    yaw = column_or_nan(navi, 'yaw');
    logtime = column_or_nan(navi, 'logtime');

    valid = lat > 20 & lat < 25 & lon > 110 & lon < 116 & ~isnan(logtime);
    if ~any(valid)
        fprintf('Skipping %s (no valid trajectory points)\n', file_name);
        continue;
    end

    idx = find(valid);
    lat = lat(idx);
    lon = lon(idx);
    alt_msl = alt_msl(idx);
    alt_agl = alt_agl(idx);
    vel_x = vel_x(idx);
    vel_y = vel_y(idx);
    vel_z = vel_z(idx);
    yaw = yaw(idx);
    logtime = logtime(idx);

    elapsed_s = (logtime - logtime(1)) ./ 1e6;
    ground_speed = hypot(vel_x, vel_y);
    point_index = (1:numel(lat)).';
    local_time = flight_start_local + seconds(elapsed_s);

    row_raw = floor(((lat - grid.min_lat) * grid.meters_per_degree_lat) ./ grid.grid_size);
    col_raw = floor(((lon - grid.min_lon) * grid.meters_per_degree_lon) ./ grid.grid_size);
    in_grid = row_raw >= 0 & row_raw < grid.n_rows & col_raw >= 0 & col_raw < grid.n_cols;
    row_idx = min(max(row_raw, 0), grid.n_rows - 1);
    col_idx = min(max(col_raw, 0), grid.n_cols - 1);
    cell_id = strcat(string(row_idx), "_", string(col_idx));

    n_points = numel(lat);
    source_col = repmat({source_folder}, n_points, 1);
    route_col = repmat({route_code}, n_points, 1);
    file_col = repmat({file_name}, n_points, 1);
    flight_col = repmat({flight_id}, n_points, 1);
    date_col = repmat({datestr(flight_start_local, 'yyyy-mm-dd')}, n_points, 1);
    datetime_local_col = cellstr(datestr(local_time, 'yyyy-mm-dd HH:MM:SS.FFF'));

    T = table( ...
        source_col, route_col, file_col, flight_col, date_col, point_index, elapsed_s, ...
        datetime_local_col, lon, lat, alt_msl, alt_agl, vel_x, vel_y, vel_z, ground_speed, yaw, ...
        row_idx, col_idx, cell_id, in_grid, ...
        'VariableNames', { ...
            'source_folder', 'route_code', 'file_name', 'flight_id', 'flight_date_local', ...
            'point_index', 'elapsed_s', 'datetime_local', 'longitude', 'latitude', ...
            'altitude_msl', 'altitude_agl', 'vel_x', 'vel_y', 'vel_z', 'ground_speed', ...
            'yaw_deg', 'grid_row', 'grid_col', 'cell_id', 'is_within_main_grid' ...
        } ...
    );

    writetable(T, tmp_csv);
    append_chunk_csv(tmp_csv, points_csv, first_chunk);
    first_chunk = false;

    n_waypoints = 0;
    if isfield(payload.FCC, 'log_waypoint')
        wp = payload.FCC.log_waypoint;
        if isfield(wp, 'lat') && isfield(wp, 'lon')
            wp_lat = double(wp.lat(:)) ./ 1e7;
            wp_lon = double(wp.lon(:)) ./ 1e7;
            n_waypoints = sum(wp_lat > 1 & wp_lon > 1);
        end
    end

    flight_end_local = flight_start_local + seconds(elapsed_s(end));
    summary_rows(end + 1, :) = { ...
        source_folder, route_code, file_name, flight_id, ...
        datestr(flight_start_local, 'yyyy-mm-dd'), ...
        datestr(flight_start_local, 'yyyy-mm-dd HH:MM:SS'), ...
        datestr(flight_end_local, 'yyyy-mm-dd HH:MM:SS'), ...
        elapsed_s(end), numel(lat), n_waypoints, ...
        lon(1), lat(1), lon(end), lat(end), ...
        alt_agl(1), alt_agl(end), median(ground_speed(~isnan(ground_speed))) ...
    };

    fprintf('  [%d/%d] %s -> %d valid points\n', i, numel(mat_files), file_name, n_points);
end

summary_table = cell2table(summary_rows, 'VariableNames', summary_header);
writetable(summary_table, summary_csv);
fprintf('Saved follow-up points: %s\n', points_csv);
fprintf('Saved follow-up summary: %s\n', summary_csv);
end

function values = column_or_nan(struct_obj, field_name)
if isfield(struct_obj, field_name)
    values = double(struct_obj.(field_name)(:));
else
    n = numel(struct_obj.logtime);
    values = nan(n, 1);
end
end

function files = collect_mat_files(root_dir)
entries = dir(root_dir);
files = {};
for i = 1:numel(entries)
    name = entries(i).name;
    if strcmp(name, '.') || strcmp(name, '..')
        continue;
    end
    full_path = fullfile(root_dir, name);
    if entries(i).isdir
        child = collect_mat_files(full_path);
        files = [files; child]; %#ok<AGROW>
    else
        [~, ~, ext] = fileparts(name);
        if strcmpi(ext, '.mat')
            files{end + 1, 1} = full_path; %#ok<AGROW>
        end
    end
end
end

function [source_folder, route_code, file_name, flight_id, flight_start_local] = parse_file_metadata(data_root, file_path)
[~, file_name, ext] = fileparts(file_path);
file_name = [file_name ext];
relative = strrep(file_path, [data_root filesep], '');
parts = strsplit(relative, filesep);
if numel(parts) >= 3
    source_folder = parts{1};
    route_code = parts{2};
else
    source_folder = 'root_logs';
    route_code = 'unassigned';
end

[~, stem, ~] = fileparts(file_path);
flight_id = sprintf('%s__%s__%s', source_folder, route_code, stem);
flight_start_local = datetime(stem, 'InputFormat', 'yyyyMMdd-HHmmss', 'TimeZone', 'Asia/Shanghai');
end

function append_chunk_csv(chunk_csv, target_csv, first_chunk)
if first_chunk
    copyfile(chunk_csv, target_csv);
    return;
end

fid_in = fopen(chunk_csv, 'r');
fid_out = fopen(target_csv, 'a');
cleanup = onCleanup(@() close_files(fid_in, fid_out));

fgetl(fid_in); % skip header
while true
    line = fgetl(fid_in);
    if ~ischar(line)
        break;
    end
    fprintf(fid_out, '%s\n', line);
end
end

function close_files(fid_in, fid_out)
if fid_in > 0
    fclose(fid_in);
end
if fid_out > 0
    fclose(fid_out);
end
end
