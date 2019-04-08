%% Demo: Load data and make plot
% To get things up and running, a simple example loading data and making a publishable plot.
%

%% Setup
clear

tic;
printcomment = @(varargin)fprintf('%-60s %5.1fs\n',sprintf(varargin{:}),toc);

output_dir = "..\Document\";
file_prefix = "plt_demo1_";
%file_prefix = "";   % If empty, no *.eps files will be saved

% Output plot size (inches)
pltdims = [4,3];

%% Load data
load project_data

%% Make Plots
plt_name = "ReferenceImage";

figure(1)
clf
imagesc(imgref)
axis image
axis off
colorbar
if strlength(file_prefix)
    % Set figure size
    set(gcf,'Units','inches')
    pos = get(gcf,'Position');
    set(gcf,'Position',[pos(1:2),pltdims]);
    print('-r300',[output_dir+file_prefix+plt_name],'-depsc');
end


plt_name = "Data";

figure(2)
clf
imagesc([0,225-5/12],0.5*61.32*[-1 1],sinogram)
axis image
colorbar
xlabel('rotation (degrees)');
ylabel('shift (mm)');
if strlength(file_prefix)
    % Set figure size
    set(gcf,'Units','inches')
    pos = get(gcf,'Position');
    set(gcf,'Position',[pos(1:2),pltdims]);
    print('-r300',[output_dir+file_prefix+plt_name],'-depsc');
end