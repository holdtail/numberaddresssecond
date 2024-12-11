function image_processing_gui()
    % 创建图形界面窗口
    f = figure('Name', '图像处理', 'Position', [500, 500, 800, 400]);
    
    % 图像显示区域，放置在窗口的右半部分
    ax = axes(f, 'Position', [0.5, 0.1, 0.45, 0.8]);  % 修改位置，使图像显示区域在右侧
    
    % 菜单栏
    mbar = uimenu(f, 'Text', '文件');
    uimenu(mbar, 'Text', '打开图像', 'MenuSelectedFcn', @(src, event) open_image(ax));  % 传递 ax
    uimenu(mbar, 'Text', '保存图像', 'MenuSelectedFcn', @(src, event) save_image(ax));  % 传递 ax
    
    % 按钮位置调整
    button_width = 120;  % 按钮宽度
    button_height = 30;  % 按钮高度
    x_offset = 50;       % 第一个按钮的水平位置
    y_offset = 20;       % 按钮垂直方向的初始位置，距离底部的间距
    button_spacing = 10; % 按钮之间的垂直间距

    % 第一列按钮（放在左侧区域）
    uicontrol('Style', 'pushbutton', 'String', '显示直方图', 'Position', [x_offset, y_offset + 3*button_height + button_spacing, button_width, button_height], 'Callback', @(src, event) show_histogram(ax));  % 传递 ax
    uicontrol('Style', 'pushbutton', 'String', '对比度增强', 'Position', [x_offset, y_offset + 4*button_height + 2*button_spacing, button_width, button_height], 'Callback', @(src, event) enhance_contrast(ax));  % 传递 ax
    uicontrol('Style', 'pushbutton', 'String', '缩放图像', 'Position', [x_offset, y_offset + 5*button_height + 3*button_spacing, button_width, button_height], 'Callback', @(src, event) scale_image(ax));
    uicontrol('Style', 'pushbutton', 'String', '提取LBP特征', 'Position', [x_offset, y_offset + 6*button_height + 4*button_spacing, button_width, button_height], 'Callback', @(src, event) lbp_features(getimage(ax)));

    % 第二列按钮（放在左侧区域）
    uicontrol('Style', 'pushbutton', 'String', '直方图均衡化', 'Position', [x_offset + button_width + 30, y_offset + 3*button_height + button_spacing, button_width, button_height], 'Callback', @(src, event) histogram_eq(ax));  % 传递 ax
    uicontrol('Style', 'pushbutton', 'String', '旋转图像', 'Position', [x_offset + button_width + 30, y_offset + 4*button_height + 2*button_spacing, button_width, button_height], 'Callback', @(src, event) rotate_image(ax));
    uicontrol('Style', 'pushbutton', 'String', '提取HOG特征', 'Position', [x_offset + button_width + 30, y_offset + 5*button_height + 3*button_spacing, button_width, button_height], 'Callback', @(src, event) hog_features(getimage(ax)));
end


%图像读取和显示
function open_image(ax)
    [file, path] = uigetfile({'*.png;*.jpg;*.bmp', '所有图像文件'}, '选择图像');
    if file ~= 0
        img = imread(fullfile(path, file));
        imshow(img, 'Parent', ax);  % 使用传递的 ax
    end
end

%灰度直方图显示
function show_histogram(ax)
    % 显示灰度直方图
    img = getimage(ax);  % 获取当前显示的图像
    if size(img, 3) == 3
        img = rgb2gray(img);  % 转为灰度图像
    end
    [counts, binLocations] = imhist(img);
    bar(binLocations, counts, 'Parent', ax);
end

%直方图均衡化
function histogram_eq(ax)
    % 直方图均衡化
    img = getimage(ax);
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    img_eq = histogram_equalization(img);
    imshow(img_eq, 'Parent', ax);
end

%对比度增强
function enhance_contrast(ax)
    % 对比度增强
    img = getimage(ax);
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    img_contrast = contrast_enhancement(img, 'log');  % 例如，使用对数变换
    imshow(img_contrast, 'Parent', ax);
end

%图像保存
function save_image(ax)
    [file, path] = uiputfile('*.png', '保存图像');
    if file ~= 0
        img = getimage(ax);
        imwrite(img, fullfile(path, file));
    end
end

% 灰度直方图均衡化
function img_eq = histogram_equalization(img)
    % 灰度直方图均衡化
    [rows, cols] = size(img);
    hist_counts = imhist(img);
    cdf = cumsum(hist_counts) / (rows * cols);  % 累积分布函数
    img_eq = uint8(255 * cdf(double(img) + 1));  % 均衡化后的图像
end

% 对比度增强
function img_contrast = contrast_enhancement(img, type)
    % 对比度增强，type为'linear'、'log'、'exp'中的一个
    img = double(img);  % 确保输入为 double 类型

    if strcmp(type, 'linear')
        % 线性对比度增强
        img_contrast = imadjust(img);
    elseif strcmp(type, 'log')
        % 对数变换，确保输入大于零
        img_contrast = log(1 + img);  % 先加 1 避免 log(0) 出错
        img_contrast = img_contrast - min(img_contrast(:));  % 归一化到 0 - 1 范围
        img_contrast = img_contrast / max(img_contrast(:)) * 255;  % 归一化到 0 - 255 范围
        img_contrast = uint8(img_contrast);  % 转回 uint8 类型
    elseif strcmp(type, 'exp')
        % 指数变换
        c = 255 / (exp(1) - 1);
        img_contrast = uint8(c * (exp(img / 255) - 1));
    end
end

%添加噪声
function img_noise = add_noise(img, noise_type, param)
    % 添加噪声
    if strcmp(noise_type, 'salt & pepper')
        img_noise = imnoise(img, 'salt & pepper', param);
    elseif strcmp(noise_type, 'gaussian')
        img_noise = imnoise(img, 'gaussian', param(1), param(2));  % mean, var
    elseif strcmp(noise_type, 'speckle')
        img_noise = imnoise(img, 'speckle', param);
    end
end

% 使用不同算子进行边缘检测
function img_edge = edge_detection(img, method)
    % 使用不同算子进行边缘检测
    if strcmp(method, 'robert')
        img_edge = edge(img, 'Roberts');
    elseif strcmp(method, 'prewitt')
        img_edge = edge(img, 'Prewitt');
    elseif strcmp(method, 'sobel')
        img_edge = edge(img, 'Sobel');
    elseif strcmp(method, 'laplacian')
        img_edge = edge(img, 'log');
    end
end

%直方图匹配（规定化）
function img_match = histogram_matching(source_img, reference_img)
    % 对源图像和参考图像进行直方图匹配
    source_hist = imhist(source_img);
    reference_hist = imhist(reference_img);
    cdf_source = cumsum(source_hist) / numel(source_img);
    cdf_reference = cumsum(reference_hist) / numel(reference_img);
    
    % 创建查找表
    lookup_table = zeros(256, 1);
    j = 1;
    for i = 1:256
        while j < 256 && cdf_reference(j) < cdf_source(i)
            j = j + 1;
        end
        lookup_table(i) = j - 1;
    end
    
    % 应用查找表进行匹配
    img_match = uint8(lookup_table(double(source_img) + 1));
end

%图像缩放和旋转变换
function scale_image(ax)
    img = getimage(ax);
    scale_factor = inputdlg('输入缩放因子 (如 0.5 或 2)：', '缩放因子', [1, 35]);
    scale_factor = str2double(scale_factor{1});
    img_scaled = imresize(img, scale_factor);
    imshow(img_scaled, 'Parent', ax);
end

function rotate_image(ax)
    img = getimage(ax);
    angle = inputdlg('输入旋转角度 (单位：度)：', '旋转角度', [1, 35]);
    angle = str2double(angle{1});
    img_rotated = imrotate(img, angle);
    imshow(img_rotated, 'Parent', ax);
end

%空域滤波（如均值滤波、Gaussian滤波）
function img_filtered = spatial_filtering(img, filter_type)
    if strcmp(filter_type, 'mean')
        h = fspecial('average', [3 3]);
    elseif strcmp(filter_type, 'gaussian')
        h = fspecial('gaussian', [3 3], 0.5);
    end
    img_filtered = imfilter(img, h);
end

%频域滤波
function img_filtered = frequency_filtering(img)
    img_fft = fft2(double(img));
    img_fft_shifted = fftshift(img_fft);
    
    % 设计一个低通滤波器
    [rows, cols] = size(img);
    [X, Y] = meshgrid(1:cols, 1:rows);
    D = sqrt((X - cols/2).^2 + (Y - rows/2).^2);
    H = double(D <= 50);  % 设定截止频率
    
    img_fft_filtered = img_fft_shifted .* H;
    img_fft_inv = ifftshift(img_fft_filtered);
    img_filtered = uint8(abs(ifft2(img_fft_inv)));
end

%目标提取
function img_segmented = segment_image(img)
    level = graythresh(img);  % Otsu法自适应阈值
    img_segmented = imbinarize(img, level);
end

%LBP特征提取
function features = lbp_features(img)
    features = extractLBPFeatures(img, 'Upright', true);
end

%HOG特征提取
function features = hog_features(img)
    features = extractHOGFeatures(img);
end
