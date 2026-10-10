function plot_guidance_scale_sweep_matlab()
rootDir = fileparts(fileparts(fileparts(mfilename('fullpath'))));
dataPath = fullfile(rootDir, 'code', 'results', 'merged', 'merged_w_sweep_with_adg_satdrift.csv');
outDir = fullfile(rootDir, 'code', 'results', 'figures', 'guidance_scale_sweep_matlab');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

T = readtable(dataPath);
methods = {'cfg', 'adg', 'apg', 'fdg', 'rptg', 'rptg_hr'};
labels = {'CFG', 'ADG', 'APG', 'FDG', 'RPTG', 'RPTG-HR'};
colors = [
    0.00 0.00 0.00
    0.00 0.62 0.45
    0.00 0.45 0.70
    0.48 0.20 0.58
    0.84 0.15 0.16
    0.84 0.15 0.16
];
lineStyles = {'-', '--', '--', ':', '-', '--'};
markers = {'o', 'x', 's', 'd', '^', 'v'};

metricKeys = {'fid', 'clip_mean', 'saturation_mean', 'contrast_mean'};
titles = {'FID \downarrow', 'CLIP \uparrow', 'Sat.', 'Contrast \downarrow'};
ylims = [70 76; 0.340 0.358; 0.18 0.60; 0.16 0.32];
yticksCell = {[70 72 74 76], [0.340 0.345 0.350 0.355], [0.20 0.30 0.40 0.50 0.60], [0.17 0.21 0.25 0.29 0.32]};

fig = figure('Visible', 'off', 'Color', 'w', 'Units', 'inches', 'Position', [1 1 7.25 1.85]);
tl = tiledlayout(fig, 1, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
legendHandles = gobjects(numel(methods), 1);

for p = 1:numel(metricKeys)
    ax = nexttile(tl);
    hold(ax, 'on');
    box(ax, 'on');
    grid(ax, 'on');
    ax.GridColor = [0.78 0.78 0.78];
    ax.GridAlpha = 0.70;
    ax.LineWidth = 0.75;
    ax.FontName = 'Times New Roman';
    ax.FontSize = 7.2;
    ax.TickDir = 'out';
    ax.XLim = [3 15];
    ax.XTick = [3 5 7.5 10 12.5 15];
    ax.YLim = ylims(p, :);
    ax.YTick = yticksCell{p};
    title(ax, titles{p}, 'FontName', 'Times New Roman', 'FontSize', 10.5, 'FontWeight', 'normal');
    xlabel(ax, '$w$', 'Interpreter', 'latex', 'FontSize', 8.5);
    if p == 1 || p == 2 || p == 4
        text(ax, 0.98, 1.08, 'zoomed', 'Units', 'normalized', 'HorizontalAlignment', 'right', ...
            'FontName', 'Times New Roman', 'FontSize', 6.5, 'Color', [0.35 0.35 0.35]);
    end

    for m = 1:numel(methods)
        methodName = methods{m};
        displayMethod = string(T.display_method);
        rowMask = strcmp(displayMethod, methodName);
        if strcmp(methodName, 'rptg')
            rowMask = rowMask | strcmp(displayMethod, 'rptg_apg');
        elseif strcmp(methodName, 'rptg_hr')
            rowMask = rowMask | strcmp(displayMethod, 'rptg_hr_apg');
        end
        S = sortrows(T(rowMask, :), 'w');
        x = S.w;
        y = S.(metricKeys{p});
        yClip = min(max(y, ylims(p, 1)), ylims(p, 2));
        lw = 1.45;
        if m >= 5
            lw = 1.70;
        end
        h = plot(ax, x, yClip, 'Color', colors(m, :), 'LineStyle', lineStyles{m}, ...
            'Marker', markers{m}, 'MarkerSize', 3.4, 'LineWidth', lw, ...
            'MarkerFaceColor', colors(m, :), 'MarkerEdgeColor', colors(m, :));
        if p == 1
            legendHandles(m) = h;
        end
        highMask = y > ylims(p, 2);
        lowMask = y < ylims(p, 1);
        if any(highMask)
            plot(ax, x(highMask), repmat(ylims(p, 2), sum(highMask), 1), '^', ...
                'Color', colors(m, :), 'MarkerFaceColor', colors(m, :), ...
                'MarkerEdgeColor', colors(m, :), 'MarkerSize', 4.2, 'LineStyle', 'none');
        end
        if any(lowMask)
            plot(ax, x(lowMask), repmat(ylims(p, 1), sum(lowMask), 1), 'v', ...
                'Color', colors(m, :), 'MarkerFaceColor', colors(m, :), ...
                'MarkerEdgeColor', colors(m, :), 'MarkerSize', 4.2, 'LineStyle', 'none');
        end
    end
end

lgd = legend(legendHandles, labels, 'Orientation', 'horizontal', 'NumColumns', 6, ...
    'Box', 'off', 'FontName', 'Times New Roman', 'FontSize', 8.0);
lgd.Layout.Tile = 'north';

outPdf = fullfile(outDir, 'fig_guidance_scale_sweep_1x4_matlab.pdf');
outPng = fullfile(outDir, 'fig_guidance_scale_sweep_1x4_matlab.png');
try
    exportgraphics(fig, outPdf, 'ContentType', 'vector');
    exportgraphics(fig, outPng, 'Resolution', 300);
catch
    print(fig, outPdf, '-dpdf', '-painters');
    print(fig, outPng, '-dpng', '-r300');
end
close(fig);
fprintf('%s\n%s\n', outPdf, outPng);
end
