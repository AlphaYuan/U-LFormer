clear all;
clc;
num_exp=12
num_fov=30
for i =1:num_exp
    for j =1:num_fov
        public_data_path = sprintf('public_data_challenge_v0/track_1/exp_%d/', i-1);
        tif_data_path = sprintf('for_track_fov_%d.tiff', j-1)
        vip_index_path = sprintf('for_vip_index_fov_%d.csv', j-1)
        tiff_path = strcat(public_data_path , tif_data_path)
        vip_index = strcat(public_data_path , vip_index_path)
        trackmate_for_vip(tiff_path, vip_index,public_data_path,j)
    end
end

%tiff_path = 'C:\Users\92036\PycharmProjects\data_track\public_data_validation_v1\track_1\exp_0\for_track_fov_0.tiff';
%vip_index=  'C:\Users\92036\PycharmProjects\data_track\public_data_validation_v1\track_1\exp_0\for_vip_index_fov_0.csv'



% x = 42;
% str = sprintf('The value of x is %d', x);
% disp(str);



%trackmate_for_vip(tiff_path, vip_index,fov)