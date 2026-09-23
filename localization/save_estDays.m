function save_estDays
rng(12);

script_dir = fileparts(mfilename('fullpath'));
data_root = fileparts(script_dir);

sensors = {'F85','F2B','ECE','E2D','F99'};
year_val = 2024;
mat_path_temp = fullfile(data_root, 'errormaps', 'errormaps_ALL_20241029_to_20241129_COMBINED.mat');

lon_min = -103; lon_max = -97;
lat_min = 16; lat_max = 30;

temp_clip_mode = 'cap';
temp_cap_value = 300;
temp_q_hi = 0.98;
temp_log_squeeze = false;
temp_alpha_tau = 1.45;
temp_tau_min = 0.20;
topk_results = 20;
ml_thresh_ratio = 0.75;

Np = 100000; % 100000
beta = 5.0;
eps_w = 1e-12;
state_bbox = [lon_min lon_max lat_min lat_max];
sig_p = [0.35 0.35]; 
sig_v = [0.25 0.25];

gt_long_start = -100.69574030606684;
gt_lat_start = 26.8028056;
gt_start = [gt_long_start, gt_lat_start];

gt_long_end = -100.26292687534539;
gt_lat_end = 19.59240561405893;
    gt_end = [gt_long_end, gt_lat_end];

force_terminal_anchor = true;
anchor_date_map = containers.Map();
anchor_date_map('F85') = datetime(year_val,11,24);
anchor_date_map('F2B') = datetime(year_val,11,11);
anchor_date_map('ECE') = datetime(year_val,11,22);
anchor_date_map('E2D') = datetime(year_val,11,23);
anchor_date_map('F99') = datetime(year_val,11,19);

plot_sample_N = Np;
eps_margin = 1e-5;
B_trajs = 250;
inflate = 1.0;

out_root_fwd = fullfile(script_dir, 'pf_estimates_all_sensors');
if ~exist(out_root_fwd,'dir'), mkdir(out_root_fwd); end

S = load(mat_path_temp,'AllSensors');
AllSensors = S.AllSensors;
ALL_FWD = struct();

for si = 1:numel(sensors)
    sensorId = sensors{si};
    
    if ~isfield(AllSensors, sensorId)
        fprintf('[%s] Not found in AllSensors. Skipping.\n', sensorId);
        continue
    end
    
    [loc_folder, results_folder] = get_sensor_folders(data_root, sensorId);
    if ~exist(loc_folder,'dir') || ~exist(results_folder,'dir')
        fprintf('[%s] Missing loc/results folders. Skipping.\n', sensorId);
        continue
    end
    
    PT = AllSensors.(sensorId);
    latT = PT.lat_grid(:);
    lonT = PT.lon_grid(:);
    Zall = PT.Z;
    daysT = to_datetime_array(PT.days);
    
    pf_date_list = unique(dateshift(daysT(:),'start','day'));
    [pf_comb_list, pf_comb_lon_list, pf_comb_lat_list, pf_comb_dates, ~, ~, ...
        ml_lon_topk, ml_lat_topk, ml_score_topk, ...
        ml_lon_thresh, ml_lat_thresh, ml_score_thresh, ...
        ml_thresh_value, ml_max_score, ml_thresh_count] = collect_combined_maps( ...
        loc_folder, results_folder, latT, lonT, Zall, daysT, pf_date_list, ...
        temp_clip_mode, temp_q_hi, temp_cap_value, temp_log_squeeze, temp_alpha_tau, temp_tau_min, ...
        lat_min, lat_max, lon_min, lon_max, topk_results, ml_thresh_ratio);
    if isempty(pf_comb_dates)
        fprintf('[%s] No PF days collected from errormaps. Skipping.\n', sensorId);
        continue
    end
    
    store_dir = fullfile(out_root_fwd, sensorId, 'pf_store');
    if ~exist(store_dir,'dir'), mkdir(store_dir); end

    FWD_only = run_pf_with_traces(pf_comb_list, pf_comb_lon_list, pf_comb_lat_list, pf_comb_dates, ...
        gt_start, ...
        Np, beta, eps_w, state_bbox, sig_p, sig_v, plot_sample_N, ...
        eps_margin, store_dir, sprintf('%s_PF_ALL', sensorId), topk_results);
    [pf_heatmap_prob, ml_heatmap_prob, pf_over_ml_prob] = score_pf_estimates_on_combined_maps( ...
        pf_comb_list, pf_comb_lon_list, pf_comb_lat_list, FWD_only.lon, FWD_only.lat, ml_max_score);

    OUT_FWD = struct();
    OUT_FWD.sensorId = sensorId;
    OUT_FWD.dates = pf_comb_dates(:);
    OUT_FWD.lon = FWD_only.lon(:);
    OUT_FWD.lat = FWD_only.lat(:);
    OUT_FWD.lon_std = FWD_only.lon_std(:);
    OUT_FWD.lat_std = FWD_only.lat_std(:);
    OUT_FWD.lon_topk = FWD_only.lon_topk;
    OUT_FWD.lat_topk = FWD_only.lat_topk;
    OUT_FWD.weight_topk = FWD_only.weight_topk;
    OUT_FWD.lon_top5 = FWD_only.lon_topk(:,1:min(5, size(FWD_only.lon_topk,2)));
    OUT_FWD.lat_top5 = FWD_only.lat_topk(:,1:min(5, size(FWD_only.lat_topk,2)));
    OUT_FWD.weight_top5 = FWD_only.weight_topk(:,1:min(5, size(FWD_only.weight_topk,2)));
    OUT_FWD.combined_ml_lon = ml_lon_topk(:,1);
    OUT_FWD.combined_ml_lat = ml_lat_topk(:,1);
    OUT_FWD.combined_ml_lon_topk = ml_lon_topk;
    OUT_FWD.combined_ml_lat_topk = ml_lat_topk;
    OUT_FWD.combined_ml_score_topk = ml_score_topk;
    OUT_FWD.combined_ml_lon_top5 = ml_lon_topk(:,1:min(5, size(ml_lon_topk,2)));
    OUT_FWD.combined_ml_lat_top5 = ml_lat_topk(:,1:min(5, size(ml_lat_topk,2)));
    OUT_FWD.combined_ml_score_top5 = ml_score_topk(:,1:min(5, size(ml_score_topk,2)));
    OUT_FWD.combined_ml_lon_thresh = ml_lon_thresh;
    OUT_FWD.combined_ml_lat_thresh = ml_lat_thresh;
    OUT_FWD.combined_ml_score_thresh = ml_score_thresh;
    OUT_FWD.combined_ml_thresh_value = ml_thresh_value(:);
    OUT_FWD.combined_ml_max_score = ml_max_score(:);
    OUT_FWD.combined_ml_thresh_count = ml_thresh_count(:);
    OUT_FWD.combined_ml_thresh_ratio = ml_thresh_ratio;
    OUT_FWD.pf_heatmap_prob = pf_heatmap_prob(:);
    OUT_FWD.ml_heatmap_prob = ml_heatmap_prob(:);
    OUT_FWD.pf_over_ml_prob = pf_over_ml_prob(:);

    if force_terminal_anchor
        key = upper(sensorId);
        if isKey(anchor_date_map, key)
            anchor_dt = dateshift(anchor_date_map(key), 'start', 'day');
            out_days = dateshift(OUT_FWD.dates(:), 'start', 'day');
            idx_anchor = find(out_days == anchor_dt, 1, 'last');
            if isempty(idx_anchor)
                fprintf('[%s] Anchor date %s not found. Keeping PF output unchanged.\n', ...
                    sensorId, datetime_to_ymd(anchor_dt));
            else
                keep_idx = 1:idx_anchor;
                OUT_FWD.dates = OUT_FWD.dates(keep_idx);
                OUT_FWD.lon = OUT_FWD.lon(keep_idx);
                OUT_FWD.lat = OUT_FWD.lat(keep_idx);
                OUT_FWD.lon_std = OUT_FWD.lon_std(keep_idx);
                OUT_FWD.lat_std = OUT_FWD.lat_std(keep_idx);
                OUT_FWD.lon_topk = OUT_FWD.lon_topk(keep_idx,:);
                OUT_FWD.lat_topk = OUT_FWD.lat_topk(keep_idx,:);
                OUT_FWD.weight_topk = OUT_FWD.weight_topk(keep_idx,:);
                OUT_FWD.lon_top5 = OUT_FWD.lon_top5(keep_idx,:);
                OUT_FWD.lat_top5 = OUT_FWD.lat_top5(keep_idx,:);
                OUT_FWD.weight_top5 = OUT_FWD.weight_top5(keep_idx,:);
                OUT_FWD.combined_ml_lon = OUT_FWD.combined_ml_lon(keep_idx);
                OUT_FWD.combined_ml_lat = OUT_FWD.combined_ml_lat(keep_idx);
                OUT_FWD.combined_ml_lon_topk = OUT_FWD.combined_ml_lon_topk(keep_idx,:);
                OUT_FWD.combined_ml_lat_topk = OUT_FWD.combined_ml_lat_topk(keep_idx,:);
                OUT_FWD.combined_ml_score_topk = OUT_FWD.combined_ml_score_topk(keep_idx,:);
                OUT_FWD.combined_ml_lon_top5 = OUT_FWD.combined_ml_lon_top5(keep_idx,:);
                OUT_FWD.combined_ml_lat_top5 = OUT_FWD.combined_ml_lat_top5(keep_idx,:);
                OUT_FWD.combined_ml_score_top5 = OUT_FWD.combined_ml_score_top5(keep_idx,:);
                OUT_FWD.combined_ml_lon_thresh = OUT_FWD.combined_ml_lon_thresh(keep_idx);
                OUT_FWD.combined_ml_lat_thresh = OUT_FWD.combined_ml_lat_thresh(keep_idx);
                OUT_FWD.combined_ml_score_thresh = OUT_FWD.combined_ml_score_thresh(keep_idx);
                OUT_FWD.combined_ml_thresh_value = OUT_FWD.combined_ml_thresh_value(keep_idx);
                OUT_FWD.combined_ml_max_score = OUT_FWD.combined_ml_max_score(keep_idx);
                OUT_FWD.combined_ml_thresh_count = OUT_FWD.combined_ml_thresh_count(keep_idx);
                OUT_FWD.pf_heatmap_prob = OUT_FWD.pf_heatmap_prob(keep_idx);
                OUT_FWD.ml_heatmap_prob = OUT_FWD.ml_heatmap_prob(keep_idx);
                OUT_FWD.pf_over_ml_prob = OUT_FWD.pf_over_ml_prob(keep_idx);
            end
        else
            fprintf('[%s] No anchor date configured. Keeping PF output unchanged.\n', sensorId);
        end
    end

    OUT_FWD.mean_pf_over_ml_prob = mean(OUT_FWD.pf_over_ml_prob, 'omitnan');
    OUT_FWD.median_pf_over_ml_prob = median(OUT_FWD.pf_over_ml_prob, 'omitnan');
    OUT_FWD.distance_to_reserve_km = haversine_km( ...
        OUT_FWD.lat, OUT_FWD.lon, gt_end(2), gt_end(1));
    OUT_FWD.final_distance_to_reserve_km = OUT_FWD.distance_to_reserve_km(end);
    ALL_FWD.(sensorId) = OUT_FWD;

    sensor_out_dir_fwd = fullfile(out_root_fwd, sensorId);
    if ~exist(sensor_out_dir_fwd,'dir'), mkdir(sensor_out_dir_fwd); end
    save(fullfile(sensor_out_dir_fwd, sprintf('pf_est_%s.mat', sensorId)), 'OUT_FWD', '-v7.3');
    T_fwd = table(OUT_FWD.dates, OUT_FWD.lat, OUT_FWD.lon, OUT_FWD.lat_std, OUT_FWD.lon_std, ...
        'VariableNames', {'date','lat','lon','lat_std','lon_std'});
    T_fwd.ml_lat = OUT_FWD.combined_ml_lat;
    T_fwd.ml_lon = OUT_FWD.combined_ml_lon;
    T_fwd.distance_to_reserve_km = OUT_FWD.distance_to_reserve_km;
    T_fwd.ml_max_score = OUT_FWD.combined_ml_max_score;
    T_fwd.ml_thresh_value = OUT_FWD.combined_ml_thresh_value;
    T_fwd.ml_thresh_count = OUT_FWD.combined_ml_thresh_count;
    T_fwd.pf_heatmap_prob = OUT_FWD.pf_heatmap_prob;
    T_fwd.ml_heatmap_prob = OUT_FWD.ml_heatmap_prob;
    T_fwd.pf_over_ml_prob = OUT_FWD.pf_over_ml_prob;
    for topk = 1:size(OUT_FWD.lat_topk,2)
        T_fwd.(sprintf('lat_top%d', topk)) = OUT_FWD.lat_topk(:,topk);
        T_fwd.(sprintf('lon_top%d', topk)) = OUT_FWD.lon_topk(:,topk);
        T_fwd.(sprintf('weight_top%d', topk)) = OUT_FWD.weight_topk(:,topk);
        T_fwd.(sprintf('ml_lat_top%d', topk)) = OUT_FWD.combined_ml_lat_topk(:,topk);
        T_fwd.(sprintf('ml_lon_top%d', topk)) = OUT_FWD.combined_ml_lon_topk(:,topk);
        T_fwd.(sprintf('ml_score_top%d', topk)) = OUT_FWD.combined_ml_score_topk(:,topk);
    end
    for topk = 1:size(OUT_FWD.lat_top5,2)
        T_fwd.(sprintf('lat_top%d', topk)) = OUT_FWD.lat_top5(:,topk);
        T_fwd.(sprintf('lon_top%d', topk)) = OUT_FWD.lon_top5(:,topk);
        T_fwd.(sprintf('weight_top%d', topk)) = OUT_FWD.weight_top5(:,topk);
    end
    writetable(T_fwd, fullfile(sensor_out_dir_fwd, sprintf('pf_est_%s.csv', sensorId)));
    
    fprintf('[%s] Saved %d PF days. First=%s Last=%s\n', sensorId, numel(OUT_FWD.dates), ...
        datetime_to_ymd(OUT_FWD.dates(1)), datetime_to_ymd(OUT_FWD.dates(end)));
    fprintf('[%s] PF/ML prob mean=%.4f median=%.4f. Final reserve distance=%.2f km on %s.\n', ...
        sensorId, OUT_FWD.mean_pf_over_ml_prob, OUT_FWD.median_pf_over_ml_prob, ...
        OUT_FWD.final_distance_to_reserve_km, datetime_to_ymd(OUT_FWD.dates(end)));
end

plot_pf_over_ml_prob_all_sensors(ALL_FWD, sensors, out_root_fwd);
plot_distance_to_reserve_all_sensors(ALL_FWD, sensors, out_root_fwd);
print_summary_table(ALL_FWD, sensors);
save(fullfile(out_root_fwd, 'pf_est_ALL_SENSORS.mat'), 'ALL_FWD', '-v7.3');
end


%% ========================= HELPER FUNCTIONS =========================

function [loc_folder, results_folder] = get_sensor_folders(data_root, sensorId)
    loc_folder = fullfile(data_root, 'lightmaps', sprintf('%s_Final', sensorId));
    results_folder = fullfile(data_root, 'lightmaps', sprintf('Result_%s_Final', sensorId), 'results');
end

function [lat_grid, lon_grid, light_map] = extract_light_map(loc_data, res_data)
lat_grid = [];
lon_grid = [];
light_map = [];

raw = res_data.results;
raw = squeeze(raw);
if ~ismatrix(raw) || isempty(raw)
    return
end

grid_candidates = {};
if isfield(loc_data,'lat_grid') && isfield(loc_data,'long_grid')
    grid_candidates(end+1,:) = {loc_data.lat_grid(:), loc_data.long_grid(:)}; 
end
if isfield(res_data,'lat_grid') && isfield(res_data,'long_grid')
    grid_candidates(end+1,:) = {res_data.lat_grid(:), res_data.long_grid(:)}; 
end
if isempty(grid_candidates)
    return
end

for ii = 1:size(grid_candidates,1)
    lat_try = grid_candidates{ii,1};
    lon_try = grid_candidates{ii,2};
    [light_try, ok] = align_light_results(raw, lat_try, lon_try);
    if ok
        lat_grid = lat_try;
        lon_grid = lon_try;
        light_map = light_try;
        break
    end
end

if isempty(light_map)
    return
end

if numel(lat_grid) > 1 && any(diff(lat_grid) < 0)
    lat_grid = flipud(lat_grid);
    light_map = flipud(light_map);
end

if numel(lon_grid) > 1 && any(diff(lon_grid) < 0)
    lon_grid = flipud(lon_grid);
    light_map = fliplr(light_map);
end

light_map(~isfinite(light_map)) = 0;
light_map(light_map < 0) = 0;
vmax = max(light_map(:));
if ~isfinite(vmax) || vmax <= 0
    lat_grid = [];
    lon_grid = [];
    light_map = [];
    return
end
light_map = light_map / vmax;
end

function [light_map, ok] = align_light_results(raw, lat_grid, lon_grid)
light_map = [];
ok = false;

nLat = numel(lat_grid);
nLon = numel(lon_grid);
sz = size(raw);

if isequal(sz, [nLat, nLon])
    light_map = double(raw);
    ok = true;
elseif isequal(sz, [nLon, nLat])
    light_map = double(raw.');
    ok = true;
elseif numel(raw) == nLat * nLon
    light_map = reshape(double(raw), nLat, nLon);
    ok = true;
end
end

function [comb_list, comb_lon_list, comb_lat_list, comb_dates, light_list, temp_list, ...
    ml_lon_topk, ml_lat_topk, ml_score_topk, ...
    ml_lon_thresh, ml_lat_thresh, ml_score_thresh, ...
    ml_thresh_value, ml_max_score, ml_thresh_count] = collect_combined_maps( ...
    loc_folder, results_folder, latT, lonT, Zall, daysT, date_list, ...
    temp_clip_mode, temp_q_hi, temp_cap_value, temp_log_squeeze, temp_alpha_tau, temp_tau_min, ...
    lat_min, lat_max, lon_min, lon_max, topk_results, ml_thresh_ratio)

comb_list = {};
comb_lon_list = {};
comb_lat_list = {};
comb_dates = datetime.empty(0,1);
light_list = {};
temp_list = {};
ml_lon_topk = nan(0,topk_results);
ml_lat_topk = nan(0,topk_results);
ml_score_topk = nan(0,topk_results);
ml_lon_thresh = cell(0,1);
ml_lat_thresh = cell(0,1);
ml_score_thresh = cell(0,1);
ml_thresh_value = nan(0,1);
ml_max_score = nan(0,1);
ml_thresh_count = nan(0,1);

daysT_day = dateshift(daysT(:),'start','day');

for dt = reshape(date_list, 1, [])
    m = month(dt); d = day(dt);
    loc_filename = sprintf('_Month_%d_Day_%d.mat', m, d);
    results_filename = sprintf('light__Month_%d_Day_%d.mat', m, d);

    loc_path = fullfile(loc_folder, loc_filename);
    res_path = fullfile(results_folder, results_filename);

    if ~exist(loc_path,'file') || ~exist(res_path,'file')
        continue
    end

    idxT = find(daysT_day == dateshift(dt,'start','day'), 1);
    if isempty(idxT)
        continue
    end

    loc_data = load(loc_path);
    res_data = load(res_path);
    if ~isfield(res_data,'results')
        continue
    end

    [latL, lonL, light_coarse] = extract_light_map(loc_data, res_data);
    if isempty(light_coarse)
        continue
    end

    Z_temp = Zall(:,:,idxT);
    Tprob_coarse = temp_to_prob(Z_temp, temp_clip_mode, temp_q_hi, temp_cap_value, ...
                                temp_log_squeeze, temp_alpha_tau, temp_tau_min);

    [lat_sub, lon_sub, light_intp, T_intp] = project_light_onto_temp_grid( ...
        latT, lonT, latL, lonL, light_coarse, Tprob_coarse, ...
        lat_min, lat_max, lon_min, lon_max);
    if isempty(light_intp) || isempty(T_intp)
        continue
    end

    C_sub = light_intp .* T_intp;
    if any(isfinite(C_sub(:)) & C_sub(:) > 0) && ~isempty(lat_sub) && ~isempty(lon_sub)
        comb_list{end+1} = C_sub; 
        comb_lon_list{end+1} = lon_sub; 
        comb_lat_list{end+1} = lat_sub; 
        comb_dates(end+1) = dt; 
        light_list{end+1} = light_intp; 
        temp_list{end+1} = T_intp; 
        [lon_topk, lat_topk, score_topk] = top_map_candidates(C_sub, lon_sub, lat_sub, topk_results);
        [lon_thresh, lat_thresh, score_thresh, thresh_value, max_score] = threshold_map_candidates( ...
            C_sub, lon_sub, lat_sub, ml_thresh_ratio);
        ml_lon_topk(end+1,:) = lon_topk; 
        ml_lat_topk(end+1,:) = lat_topk; 
        ml_score_topk(end+1,:) = score_topk; 
        ml_lon_thresh{end+1,1} = lon_thresh; 
        ml_lat_thresh{end+1,1} = lat_thresh; 
        ml_score_thresh{end+1,1} = score_thresh; 
        ml_thresh_value(end+1,1) = thresh_value; 
        ml_max_score(end+1,1) = max_score; 
        ml_thresh_count(end+1,1) = numel(score_thresh); 
    end
end
end

function [lon_topk, lat_topk, score_topk] = top_map_candidates(C_sub, lon_sub, lat_sub, topk)
lon_topk = nan(1, topk);
lat_topk = nan(1, topk);
score_topk = nan(1, topk);

if isempty(C_sub) || isempty(lon_sub) || isempty(lat_sub)
    return
end

[LonGrid, LatGrid] = meshgrid(lon_sub, lat_sub);
scores = C_sub(:);
lon_vals = LonGrid(:);
lat_vals = LatGrid(:);

good = isfinite(scores) & isfinite(lon_vals) & isfinite(lat_vals);
if ~any(good)
    return
end

scores = scores(good);
lon_vals = lon_vals(good);
lat_vals = lat_vals(good);

[scores_sorted, idx] = sort(scores, 'descend');
n_keep = min(topk, numel(idx));
idx = idx(1:n_keep);
scores_sorted = scores_sorted(1:n_keep);

lon_topk(1:n_keep) = lon_vals(idx);
lat_topk(1:n_keep) = lat_vals(idx);
score_topk(1:n_keep) = scores_sorted;
end

function [lon_keep, lat_keep, score_keep, thresh_value, max_score] = threshold_map_candidates( ...
    C_sub, lon_sub, lat_sub, thresh_ratio)
lon_keep = [];
lat_keep = [];
score_keep = [];
thresh_value = nan;
max_score = nan;

if isempty(C_sub) || isempty(lon_sub) || isempty(lat_sub)
    return
end

[LonGrid, LatGrid] = meshgrid(lon_sub, lat_sub);
scores = C_sub(:);
lon_vals = LonGrid(:);
lat_vals = LatGrid(:);

good = isfinite(scores) & isfinite(lon_vals) & isfinite(lat_vals);
if ~any(good)
    return
end

scores = scores(good);
lon_vals = lon_vals(good);
lat_vals = lat_vals(good);

max_score = max(scores);
if ~isfinite(max_score)
    return
end

thresh_value = thresh_ratio * max_score;
keep = scores >= thresh_value;
if ~any(keep)
    return
end

score_keep = scores(keep);
lon_keep = lon_vals(keep);
lat_keep = lat_vals(keep);

[score_keep, idx] = sort(score_keep, 'descend');
lon_keep = lon_keep(idx);
lat_keep = lat_keep(idx);
end

function [lat_sub, lon_sub, light_on_temp, temp_sub] = project_light_onto_temp_grid( ...
    latT, lonT, latL, lonL, light_coarse, Tprob_coarse, ...
    lat_min, lat_max, lon_min, lon_max)

lat_sub = [];
lon_sub = [];
light_on_temp = [];
temp_sub = [];

lat_lo = max([lat_min, min(latT), min(latL)]);
lat_hi = min([lat_max, max(latT), max(latL)]);
lon_lo = max([lon_min, min(lonT), min(lonL)]);
lon_hi = min([lon_max, max(lonT), max(lonL)]);

if ~(lat_lo < lat_hi && lon_lo < lon_hi)
    return
end

mask_lat = latT >= lat_lo & latT <= lat_hi;
mask_lon = lonT >= lon_lo & lonT <= lon_hi;
lat_sub = latT(mask_lat);
lon_sub = lonT(mask_lon);

if isempty(lat_sub) || isempty(lon_sub)
    lat_sub = [];
    lon_sub = [];
    return
end

[LonL, LatL] = meshgrid(lonL, latL);
[LonSub, LatSub] = meshgrid(lon_sub, lat_sub);
light_on_temp = interp2(LonL, LatL, light_coarse, LonSub, LatSub, 'linear', 0);
light_on_temp(~isfinite(light_on_temp)) = 0;
light_on_temp(light_on_temp < 0) = 0;

vmaxL = max(light_on_temp(:));
if ~isfinite(vmaxL) || vmaxL <= 0
    lat_sub = [];
    lon_sub = [];
    light_on_temp = [];
    return
end
light_on_temp = light_on_temp / vmaxL;

temp_sub = Tprob_coarse(mask_lat, mask_lon);
temp_sub(~isfinite(temp_sub)) = 0;
temp_sub(temp_sub < 0) = 0;

vmaxT = max(temp_sub(:));
if ~isfinite(vmaxT) || vmaxT <= 0
    lat_sub = [];
    lon_sub = [];
    light_on_temp = [];
    temp_sub = [];
    return
end
temp_sub = temp_sub / (vmaxT + eps);
end

function [pf_prob, ml_prob, pf_over_ml] = score_pf_estimates_on_combined_maps( ...
    map_list, lon_list, lat_list, pf_lon, pf_lat, ml_max_score)

K = numel(map_list);
pf_prob = nan(K,1);
ml_prob = nan(K,1);
pf_over_ml = nan(K,1);

pf_lon = pf_lon(:);
pf_lat = pf_lat(:);
ml_max_score = ml_max_score(:);

for k = 1:K
    if k > numel(pf_lon) || k > numel(pf_lat)
        continue
    end
    if ~isfinite(pf_lon(k)) || ~isfinite(pf_lat(k))
        continue
    end
    if k > numel(lon_list) || k > numel(lat_list) || isempty(map_list{k})
        continue
    end

    Ck = map_list{k};
    LONS = lon_list{k};
    LATS = lat_list{k};
    if isempty(Ck) || isempty(LONS) || isempty(LATS)
        continue
    end

    [LonGk, LatGk] = meshgrid(LONS, LATS);
    pf_prob(k) = interp2(LonGk, LatGk, Ck, pf_lon(k), pf_lat(k), 'linear', 0);
    if k <= numel(ml_max_score)
        ml_prob(k) = ml_max_score(k);
    end

    if isfinite(pf_prob(k)) && isfinite(ml_prob(k)) && ml_prob(k) > 0
        pf_over_ml(k) = pf_prob(k) / ml_prob(k);
    end
end
end

function plot_pf_over_ml_prob_all_sensors(ALL_FWD, sensor_order, out_root_fwd)
if isempty(fieldnames(ALL_FWD))
    return
end

f = figure( ...
    'Color','w', ...
    'Visible','off', ...
    'Name','PF prob over ML prob', ...
    'Units','pixels', ...
    'Position',[100 100 1200 720]);
ax = axes('Parent', f);
hold(ax, 'on');
grid(ax, 'on');

colors = lines(max(1, numel(sensor_order)));
handles = gobjects(0);
labels = {};
avg_labels = {};
avg_colors = zeros(0,3);
y_all = [];

for i = 1:numel(sensor_order)
    sensorId = sensor_order{i};
    if ~isfield(ALL_FWD, sensorId)
        continue
    end

    OUT = ALL_FWD.(sensorId);
    if ~isfield(OUT, 'dates') || ~isfield(OUT, 'pf_over_ml_prob')
        continue
    end

    dt = OUT.dates(:);
    if isnumeric(dt)
        dt = datetime(dt, 'ConvertFrom', 'datenum');
    end

    y = OUT.pf_over_ml_prob(:);
    n = min(numel(dt), numel(y));
    dt = dt(1:n);
    y = y(1:n);
    good = ~isnat(dt) & isfinite(y);
    if ~any(good)
        continue
    end

    h = plot(ax, dt(good), y(good), ...
        'LineWidth',2.25, ...
        'Marker','o', ...
        'MarkerSize',5.5, ...
        'Color',colors(i,:));
    handles(end+1) = h; 
    labels{end+1} = sensorId; 
    if isfield(OUT, 'mean_pf_over_ml_prob') && isfinite(OUT.mean_pf_over_ml_prob)
        avg_ratio = OUT.mean_pf_over_ml_prob;
    else
        avg_ratio = mean(y(good), 'omitnan');
    end
    avg_labels{end+1} = sprintf('%s avg = %.3f', sensorId, avg_ratio); 
    avg_colors(end+1,:) = colors(i,:); 
    y_all = [y_all; y(good)]; 
end

if isempty(handles)
    close(f);
    return
end

ylabel(ax, 'PF prob / ML prob');
xlabel(ax, 'Date');
title(ax, 'PF Estimate Probability Relative to ML Maximum');
legend(ax, handles, labels, 'Location','northwest');
set(ax, 'FontSize',16);

y_hi = max([1.05; y_all(:)], [], 'omitnan');
ylim(ax, [0, y_hi * 1.05]);

ax_avg = axes('Parent', f, ...
    'Position', get(ax, 'Position'), ...
    'Color','none', ...
    'Visible','off');
hold(ax_avg, 'on');
avg_handles = gobjects(numel(avg_labels),1);
for i = 1:numel(avg_labels)
    avg_handles(i) = plot(ax_avg, NaN, NaN, ...
        'LineWidth',2.25, ...
        'Marker','o', ...
        'MarkerSize',5.5, ...
        'Color',avg_colors(i,:));
end
avg_legend = legend(ax_avg, avg_handles, avg_labels, 'Location','northeast');
title(avg_legend, 'Average ratio');
set(avg_legend, 'FontSize',12);

out_file = fullfile(out_root_fwd, 'pf_over_ml_prob_ALL_SENSORS.png');
print(f, out_file, '-dpng', '-r600');
close(f);
end

function plot_distance_to_reserve_all_sensors(ALL_FWD, sensor_order, out_root_fwd)
if isempty(fieldnames(ALL_FWD))
    return
end

f = figure( ...
    'Color','w', ...
    'Visible','off', ...
    'Name','PF distance to reserve', ...
    'Units','pixels', ...
    'Position',[100 100 1200 720]);
ax = axes('Parent', f);
hold(ax, 'on');
grid(ax, 'on');

colors = lines(max(1, numel(sensor_order)));
handles = gobjects(0);
labels = {};
y_all = [];

for i = 1:numel(sensor_order)
    sensorId = sensor_order{i};
    if ~isfield(ALL_FWD, sensorId)
        continue
    end

    OUT = ALL_FWD.(sensorId);
    if ~isfield(OUT, 'dates') || ~isfield(OUT, 'distance_to_reserve_km')
        continue
    end

    dt = OUT.dates(:);
    if isnumeric(dt)
        dt = datetime(dt, 'ConvertFrom', 'datenum');
    end

    y = OUT.distance_to_reserve_km(:);
    n = min(numel(dt), numel(y));
    dt = dt(1:n);
    y = y(1:n);
    good = ~isnat(dt) & isfinite(y);
    if ~any(good)
        continue
    end

    h = plot(ax, dt(good), y(good), ...
        'LineWidth',2.25, ...
        'Marker','o', ...
        'MarkerSize',5.5, ...
        'Color',colors(i,:));
    handles(end+1) = h; 
    labels{end+1} = sensorId; 
    y_all = [y_all; y(good)]; 
end

if isempty(handles)
    close(f);
    return
end

ylabel(ax, 'Distance to reserve (km)');
xlabel(ax, 'Date');
title(ax, 'PF Estimate Distance to Reserve');
legend(ax, handles, labels, 'Location','northeast');
set(ax, 'FontSize',16);

y_hi = max(y_all(:), [], 'omitnan');
if isfinite(y_hi) && y_hi > 0
    ylim(ax, [0, y_hi * 1.05]);
end

out_file = fullfile(out_root_fwd, 'pf_distance_to_reserve_ALL_SENSORS.png');
print(f, out_file, '-dpng', '-r600');
close(f);
end

function print_summary_table(ALL_FWD, sensor_order)
summary_sensor = {};
summary_mean = [];
summary_median = [];
summary_final_distance = [];
summary_final_date = datetime.empty(0,1);

for i = 1:numel(sensor_order)
    sensorId = sensor_order{i};
    if ~isfield(ALL_FWD, sensorId)
        continue
    end

    OUT = ALL_FWD.(sensorId);
    if ~isfield(OUT, 'dates') || isempty(OUT.dates)
        continue
    end

    summary_sensor{end+1,1} = sensorId; 
    summary_mean(end+1,1) = get_numeric_field_or_nan(OUT, 'mean_pf_over_ml_prob'); 
    summary_median(end+1,1) = get_numeric_field_or_nan(OUT, 'median_pf_over_ml_prob'); 
    summary_final_distance(end+1,1) = get_numeric_field_or_nan(OUT, 'final_distance_to_reserve_km'); 

    final_date = OUT.dates(end);
    if isnumeric(final_date)
        final_date = datetime(final_date, 'ConvertFrom', 'datenum');
    end
    final_date = dateshift(final_date, 'start', 'day');
    summary_final_date(end+1,1) = final_date; 
end

if isempty(summary_sensor)
    fprintf('\nNo sensor summary table available.\n');
    return
end

Summary = table( ...
    string(summary_sensor), ...
    summary_mean, ...
    summary_median, ...
    summary_final_distance, ...
    summary_final_date, ...
    'VariableNames', {'Sensor','PF_ML_Mean','PF_ML_Median','FinalDistanceToReserve_km','FinalDate'});
Summary.FinalDate.Format = 'yyyy-MM-dd';

fprintf('\nSummary results:\n');
disp(Summary);
end

function value = get_numeric_field_or_nan(S, field_name)
if isfield(S, field_name) && isnumeric(S.(field_name)) && ~isempty(S.(field_name))
    value = S.(field_name);
    value = value(1);
else
    value = nan;
end
end

function EST_fwd = run_pf_with_traces(map_list, lon_list, lat_list, dates_list, ...
    gt_start, Np, beta, eps_w, state_bbox, sig_p, sig_v, plot_sample_N, eps_margin, store_dir, tag, topk_results)

K = numel(map_list);
EST_fwd.lon = nan(K,1);
EST_fwd.lat = nan(K,1);
EST_fwd.lon_std = nan(K,1);
EST_fwd.lat_std = nan(K,1);
EST_fwd.lon_topk = nan(K,topk_results);
EST_fwd.lat_topk = nan(K,topk_results);
EST_fwd.weight_topk = nan(K,topk_results);

if K == 0
    return
end

if nargin < 12 || isempty(plot_sample_N), plot_sample_N = Np; end
if nargin < 13 || isempty(eps_margin), eps_margin = 1e-5; end
if nargin < 16 || isempty(topk_results), topk_results = 20; end

state_bbox = prepare_state_bbox(state_bbox, eps_margin);

store_file = fullfile(store_dir, sprintf('pf_%s.mat', tag));
if exist(store_file,'file'), delete(store_file); end
mf = matfile(store_file, 'Writable', true);
mf.Np = Np;
mf.K = K;

[x0, y0] = clip_points_to_bbox(gt_start(1), gt_start(2), state_bbox);
X = x0 * ones(Np,1);
Y = y0 * ones(Np,1);
Vx = zeros(Np,1);
Vy = zeros(Np,1);
w = ones(Np,1) / Np;

ss = min(plot_sample_N, Np);
samp_idx = randsample(Np, ss, false);

for k = 1:K
    LONS = lon_list{k};
    LATS = lat_list{k};
    Ck = map_list{k};
    [LonGk, LatGk] = meshgrid(LONS, LATS);

    if k > 1
        dtday = days(dates_list(k) - dates_list(k-1));
        if dtday > 0
            [X, Y, Vx, Vy] = propagate_one_day_cv_bounded(X, Y, Vx, Vy, dtday, sig_p, sig_v, state_bbox);
        end
    end

    Lk = interp2(LonGk, LatGk, Ck, X, Y, 'linear', 0);
    Lk = max(Lk, eps_w).^beta;
    w = w .* Lk;
    sw = sum(w);
    if sw <= 0 || ~isfinite(sw)
        w = ones(Np,1) / Np;
    else
        w = w ./ sw;
    end

    [EST_fwd.lon(k), EST_fwd.lon_std(k)] = weighted_stats(X, w);
    [EST_fwd.lat(k), EST_fwd.lat_std(k)] = weighted_stats(Y, w);
    [EST_fwd.lon_topk(k,:), EST_fwd.lat_topk(k,:), EST_fwd.weight_topk(k,:)] = top_weighted_candidates(X, Y, w, topk_results);

    mf.(['X_' num2str(k)]) = X(samp_idx);
    mf.(['Y_' num2str(k)]) = Y(samp_idx);
    mf.(['W_' num2str(k)]) = w(samp_idx);

    ind = sys_resample(w);
    X = X(ind);
    Y = Y(ind);
    Vx = Vx(ind);
    Vy = Vy(ind);
    w = ones(Np,1) / Np;
end

function [lon_topk, lat_topk, w_topk] = top_weighted_candidates(X, Y, w, topk)
    lon_topk = nan(1, topk);
    lat_topk = nan(1, topk);
    w_topk = nan(1, topk);

    good = isfinite(X) & isfinite(Y) & isfinite(w) & w >= 0;
    if ~any(good)
        return
    end

    X = X(good);
    Y = Y(good);
    w = w(good);

    [w_sorted, idx] = sort(w, 'descend');
    n_keep = min(topk, numel(idx));
    idx = idx(1:n_keep);
    w_sorted = w_sorted(1:n_keep);

    lon_topk(1:n_keep) = X(idx);
    lat_topk(1:n_keep) = Y(idx);
    w_topk(1:n_keep) = w_sorted;
end
end

function [mu, sigma] = weighted_stats(x, w)
    x = x(:);
    w = w(:);

    good = isfinite(x) & isfinite(w) & w >= 0;
    if ~any(good)
        mu = nan;
        sigma = nan;
        return
    end

    x = x(good);
    w = w(good);
    sw = sum(w);
    if sw <= 0 || ~isfinite(sw)
        mu = nan;
        sigma = nan;
        return
    end

    w = w / sw;
    mu = sum(w .* x);
    varx = sum(w .* (x - mu).^2);
    sigma = sqrt(max(varx, 0));
end


function [Xo, Yo, Vxo, Vyo] = propagate_one_day_cv_bounded(Xi, Yi, Vxi, Vyi, dtday, sig_p, sig_v, state_bbox)
    Xo = Xi; Yo = Yi; Vxo = Vxi; Vyo = Vyi;
    if ~isfinite(dtday) || dtday <= 0
        return
    end

    xmin = state_bbox(1); xmax = state_bbox(2);
    ymin = state_bbox(3); ymax = state_bbox(4);
    pending = true(size(Xi));

    % Rejection sampling draws exactly from the truncated transition.
    while any(pending)
        idx = find(pending);
        npend = numel(idx);

        Vx_try = Vxi(idx) + sig_v(1) * sqrt(dtday) * randn(npend,1);
        Vy_try = Vyi(idx) + sig_v(2) * sqrt(dtday) * randn(npend,1);
        X_try = Xi(idx) + Vx_try * dtday + sig_p(1) * sqrt(dtday) * randn(npend,1);
        Y_try = Yi(idx) + Vy_try * dtday + sig_p(2) * sqrt(dtday) * randn(npend,1);

        accept = X_try >= xmin & X_try <= xmax & Y_try >= ymin & Y_try <= ymax;
        if any(accept)
            ia = idx(accept);
            Xo(ia) = X_try(accept);
            Yo(ia) = Y_try(accept);
            Vxo(ia) = Vx_try(accept);
            Vyo(ia) = Vy_try(accept);
            pending(ia) = false;
        end
    end
end

function bbox = prepare_state_bbox(state_bbox, eps_margin)
    bbox = state_bbox(:).';
    if numel(bbox) ~= 4 || any(~isfinite(bbox)) || bbox(1) >= bbox(2) || bbox(3) >= bbox(4)
        error('Invalid state_bbox.');
    end

    mx = min(max(eps_margin, 0), 0.49 * (bbox(2) - bbox(1)));
    my = min(max(eps_margin, 0), 0.49 * (bbox(4) - bbox(3)));
    bbox = [bbox(1) + mx, bbox(2) - mx, bbox(3) + my, bbox(4) - my];
end

function [Xc, Yc] = clip_points_to_bbox(X, Y, state_bbox)
    Xc = min(max(X, state_bbox(1)), state_bbox(2));
    Yc = min(max(Y, state_bbox(3)), state_bbox(4));
end

function ind = sys_resample(w)
    N = numel(w);
    edges = min([0; cumsum(w)], 1); 
    edges(end) = 1;
    u0 = rand/N; 
    u = u0 + ((0:N-1)')/N;
    [~, ind] = histc(u, edges);
    ind = max(1, min(N, ind));
end

function P = temp_to_prob(Z, clip_mode, q_hi, cap_val, log_squeeze, alpha_tau, tau_min)
    P = nan(size(Z));
    mask = isfinite(Z); 
    if ~any(mask,'all'), return; end
    Zc = Z; 
    v = Zc(mask);
    
    switch lower(clip_mode)
        case 'quantile'
            hi = prctile(v, q_hi*100); 
            Zc(Zc > hi) = hi;
        case 'cap'
            Zc(Zc > cap_val) = cap_val;
        case 'none'
        otherwise
            error('Unknown clip_mode: %s', clip_mode);
    end
    
    if log_squeeze
        Zc(mask) = log1p(max(Zc(mask), 0));
    end
    
    s = iqr(Zc(mask)); 
    if ~isfinite(s) || s<=0
        s = std(Zc(mask),'omitnan'); 
    end
    if ~isfinite(s) || s<=0, s = 1; end
    
    tau = max(tau_min, alpha_tau * s);
    zmin = min(Zc(mask)); 
    X = -(Zc - zmin) / tau;
    P = softmax2D(X);
end

function P = softmax2D(X)
    mask = isfinite(X);
    xm = max(X(mask));
    EX = zeros(size(X));
    EX(mask) = exp(X(mask) - xm);
    Z = sum(EX(mask));
    if Z <= 0 || ~isfinite(Z)
        P = nan(size(X));
    else
        P = zeros(size(X));
        P(mask) = EX(mask)/Z;
        P(~mask) = nan;
    end
end

function d_km = haversine_km(lat1, lon1, lat2, lon2)
    earth_radius_km = 6371.0088;
    lat1 = double(lat1);
    lon1 = double(lon1);
    lat2 = double(lat2);
    lon2 = double(lon2);

    phi1 = deg2rad(lat1);
    phi2 = deg2rad(lat2);
    dphi = deg2rad(lat2 - lat1);
    dlambda = deg2rad(lon2 - lon1);

    a = sin(dphi ./ 2).^2 + cos(phi1) .* cos(phi2) .* sin(dlambda ./ 2).^2;
    a = min(max(a, 0), 1);
    d_km = earth_radius_km .* 2 .* atan2(sqrt(a), sqrt(1 - a));
end

function s = datetime_to_ymd(dt)
    if isnumeric(dt)
        dt = datetime(dt, 'ConvertFrom', 'datenum');
    end
    dt.Format = 'yyyy-MM-dd';
    s = char(string(dt));
end

function dt = to_datetime_array(x)
    if isdatetime(x)
        dt = x;
    elseif isnumeric(x)
        dt = datetime(x,'ConvertFrom','datenum');
    else
        error('Unsupported days type in PT.days.');
    end
end
