% Plot RPTG directional spectral drift with MATLAB LaTeX-rendered labels.
% Run from the repository root:
%   matlab -batch "run('code/plotting/plot_rptg_formula_energy_matlab.m')"

clear; close all; clc;

scriptDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(scriptDir));
summaryCsv = fullfile(repoRoot, 'code', 'results', 'directional_energy_summary.csv');
outputDir = fullfile(repoRoot, 'RPTG_AAAI2027_Submission', 'Figures', 'DirectionalEnergy');
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

T = readtable(summaryCsv, 'TextType', 'string');
scales = sort(unique(T.cfg_scale)).';

values = struct();
values.all_radial = zeros(size(scales));
values.all_tangential = zeros(size(scales));
values.low_radial = zeros(size(scales));
values.low_tangential = zeros(size(scales));
values.high_radial = zeros(size(scales));
values.high_tangential = zeros(size(scales));

for i = 1:numel(scales)
    w = scales(i);
    lowPos = rowValue(T, w, "low_positive_radial", "weighted_energy_mean");
    lowNeg = rowValue(T, w, "low_negative_radial", "weighted_energy_mean");
    lowTan = rowValue(T, w, "low_tangential", "weighted_energy_mean");
    highPos = rowValue(T, w, "high_positive_radial", "weighted_energy_mean");
    highNeg = rowValue(T, w, "high_negative_radial", "weighted_energy_mean");
    highTan = rowValue(T, w, "high_tangential", "weighted_energy_mean");
    lowCoupling = rowValue(T, w, "low_positive_radial", "positive_radial_coupling_mean");
    highCoupling = rowValue(T, w, "high_positive_radial", "positive_radial_coupling_mean");

    values.low_radial(i) = lowCoupling + lowPos + lowNeg;
    values.high_radial(i) = highCoupling + highPos + highNeg;
    values.low_tangential(i) = lowTan;
    values.high_tangential(i) = highTan;
    values.all_radial(i) = values.low_radial(i) + values.high_radial(i);
    values.all_tangential(i) = values.low_tangential(i) + values.high_tangential(i);
end

colors.all = [0.86, 0.10, 0.13];
colors.low = [0.92, 0.42, 0.00];
colors.high = [0.00, 0.38, 0.68];

lineWidth = 0.95;
markerSize = 2.65;

fig = figure('Color', 'w', 'Units', 'centimeters', 'Position', [3, 3, 8.8, 5.9]);
ax = axes(fig);
hold(ax, 'on');
set(ax, 'Position', [0.18, 0.18, 0.77, 0.75]);

hAllRad = plot(scales, values.all_radial, '-o', 'Color', colors.all, 'LineWidth', lineWidth, 'MarkerSize', markerSize, 'MarkerFaceColor', colors.all, 'MarkerEdgeColor', colors.all, 'DisplayName', '$\Delta X_r$');
hAllTau = plot(scales, values.all_tangential, '-d', 'Color', colors.all, 'LineWidth', lineWidth, 'MarkerSize', markerSize, 'MarkerFaceColor', colors.all, 'MarkerEdgeColor', colors.all, 'DisplayName', '$\Delta X_{\tau}$');
hLowRad = plot(scales, values.low_radial, '--o', 'Color', colors.low, 'LineWidth', lineWidth, 'MarkerSize', markerSize, 'MarkerFaceColor', colors.low, 'MarkerEdgeColor', colors.low, 'DisplayName', '$\Delta X_{r,L}$');
hLowTau = plot(scales, values.low_tangential, '--d', 'Color', colors.low, 'LineWidth', lineWidth, 'MarkerSize', markerSize, 'MarkerFaceColor', colors.low, 'MarkerEdgeColor', colors.low, 'DisplayName', '$\Delta X_{\tau,L}$');
hHighRad = plot(scales, values.high_radial, '-o', 'Color', colors.high, 'LineWidth', lineWidth, 'MarkerSize', markerSize, 'MarkerFaceColor', colors.high, 'MarkerEdgeColor', colors.high, 'DisplayName', '$\Delta X_{r,H}$');
hHighTau = plot(scales, values.high_tangential, '-d', 'Color', colors.high, 'LineWidth', lineWidth, 'MarkerSize', markerSize, 'MarkerFaceColor', colors.high, 'MarkerEdgeColor', colors.high, 'DisplayName', '$\Delta X_{\tau,H}$');

box(ax, 'off');
grid(ax, 'on');
ax.GridColor = [0.88, 0.88, 0.88];
ax.GridAlpha = 0.42;
ax.LineWidth = 0.72;
ax.FontName = 'Arial';
ax.FontSize = 7.1;
ax.TickDir = 'out';
ax.XLim = [2.9, 15.7];
ax.YLim = [0, 2.5];
ax.XTick = scales;
ax.YTick = 0:0.5:2.5;
ax.Layer = 'bottom';

xlabel(ax, 'Guidance scale $w$', 'Interpreter', 'latex', 'FontSize', 8.1);
ylabel(ax, 'Mean spectral drift term', 'Interpreter', 'latex', 'FontSize', 8.1);
text(ax, 9.55, 1.84, '$\Delta X_r$ (full)', 'Interpreter', 'latex', 'Color', colors.all, 'FontSize', 7.5, 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'BackgroundColor', [1, 1, 1], 'Margin', 0.8);
text(ax, 12.75, 1.55, '$\Delta X_{r,L}$ (low)', 'Interpreter', 'latex', 'Color', colors.low, 'FontSize', 7.5, 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'BackgroundColor', [1, 1, 1], 'Margin', 0.8);

lgd = legend(ax, [hAllRad, hLowRad, hHighRad, hAllTau, hLowTau, hHighTau], {'$\Delta X_r$', '$\Delta X_{r,L}$', '$\Delta X_{r,H}$', '$\Delta X_{\tau}$', '$\Delta X_{\tau,L}$', '$\Delta X_{\tau,H}$'}, 'Location', 'northwest', 'Interpreter', 'latex', 'NumColumns', 2);
lgd.Box = 'on';
lgd.EdgeColor = [0.58, 0.58, 0.58];
lgd.Color = [1, 1, 1];
lgd.FontSize = 6.8;
lgd.ItemTokenSize = [14, 7];

set(fig, 'Renderer', 'painters');
pdfPath = fullfile(outputDir, 'fig_six_directional_formula_terms_matlab.pdf');
pngPath = fullfile(outputDir, 'fig_six_directional_formula_terms_matlab.png');
print(fig, pngPath, '-dpng', '-r300');
try
    exportgraphics(fig, pdfPath, 'ContentType', 'vector', 'BackgroundColor', 'white');
catch exportError
    warning('Vector PDF export failed: %s', exportError.message);
    try
        print(fig, pdfPath, '-dpdf', '-painters');
    catch printError
        warning('Fallback PDF export failed: %s', printError.message);
    end
end

fprintf('Wrote %s\n', pdfPath);
fprintf('Wrote %s\n', pngPath);

function value = rowValue(T, scale, component, column)
    idx = T.cfg_scale == scale & T.component == component;
    if ~any(idx)
        error('Missing row for scale %.4g and component %s.', scale, component);
    end
    value = T.(column)(find(idx, 1, 'first'));
end
