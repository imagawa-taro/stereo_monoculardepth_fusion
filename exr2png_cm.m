depth = exrread('Depth.exr');
disp(max(depth(:)));
disp(min(depth(:)));

depth_cm = uint16(round(depth * 10000));
imwrite(depth_cm, 'Depth_cm.png');
