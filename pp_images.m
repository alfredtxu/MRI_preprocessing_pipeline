%{
Read files containing stroke lesions in nifty format, create a mask for 
the voxels that belong to a lesion in at least 5 images, and create a 
.csv file containing each voxel (1 file per row). 
Jane M. Rondina, updated on 08/09/2023
%}

addpath('/home/jrondina/spm12')

clear all;

site = 'KCH'
time_frame = 'retrospective_and_prospective'


% % img_path = '/media/jrondina/MPBRCII/a_imagepool_mr/ischaemic_stroke_seg_latest/segmentation_new_control_trained_model/';
datasets_path = '/home/jrondina/Desktop/PycharmProjects/predictive-tool/DATASETS/';

if (strcmp(site, 'UCLH') == 1)
    matchFolder = strcat(datasets_path, 'SEGMENTATIONS/RETROSPECTIVE_UCLH_SSNAP_LESIONS/');
    matchFile = strcat(matchFolder, 'matching__UCLH__SSNAP__retrospective__to__LESIONS.csv');
elseif (strcmp(site, 'KCH') == 1)
    matchFolder = strcat(datasets_path, 'SEGMENTATIONS/RETROSPECTIVE_PROSPECTIVE_KCL_SSNAP_LESIONS/')
    matchFile = strcat(matchFolder, 'matching__KCH__SSNAP__retrospective_and_prospective__to__LESIONS.csv')
else
    disp('Invalid option - accepted UCLH or KCH');
end


% % % ----------------------------------------------------------------
% % % Step 1: Create new local folder for lesions matched with clinical dataset
% % % and copy lesion files to it. Additionally, extract files and remove .gz.
% % %----------------------------------------------------------------
% % 
% % 
% % mkdir(matchFolder);
T = readtable(matchFile);

lesionsFolder = strcat(matchFolder, 'ALL/')
 
% for lsf=1:size(T,1) 
%     disp(lsf)
%     lesion_filename = T.lesionPath{lsf}
%     if (strcmp(site, 'UCLH') == 1)
%         lesion_filename_in_mounted_folder = strrep(lesion_filename,'/media/chrisfoulon/DATA1/','/media/jrondina/MPBRCII/');
%     end
%     if (strcmp(site, 'KCH') == 1) 
%         lesion_filename_in_mounted_folder = strcat('/media/jrondina/MPBRCII/Jane/KCH_LESIONS/', lesion_filename)
%     end
%     if strcmp(lesion_filename, '#conversion error') == 0
%         copyfile(lesion_filename_in_mounted_folder, lesionsFolder);  
%         [filepath,name,ext] = fileparts(lesion_filename_in_mounted_folder);    
%         gunzip(strcat(lesionsFolder, name, ext));
%         delete(strcat(lesionsFolder, name, ext));
%     end
% end
%  
% % 
% % % Read names of lesion files (with absolute path) from matching
% % csvFile = strcat(datasets_path, 'matching__SSNAP_retrospective__LESIONS_updated.csv');
% % T = readtable(csvFile);
% % 
% % T.imgValidFlag = zeros(size(T,1),1);
% % T.correctedPath = repmat({''}, size(T,1), 1); 
% % 
% % 
% % 
% %----------------------------------------------------------------
% % Step 2: Smooth lesions (FWHMV 4x4x4) and resize // currently
% % performed in smp interface, to be replaced for command line.
% %----------------------------------------------------------------
% % 
% %----------------------------------------------------------------
% % Step 3: Resize smoothed images.
% %----------------------------------------------------------------
% 
% lesion_files_list = dir(lesionsFolder);
% for i=3:size(lesion_files_list,1)
%     if (startsWith(lesion_files_list(i).name,'s_'))        
%         smoothedlesion = strcat(lesionsFolder,lesion_files_list(i).name)
%         resize_img(smoothedlesion, [4 4 4], nan(2,3));
%     end
% end
% % 
% 
% %----------------------------------------------------------------
% % Step 4: Create mask
% % Save binary mask (threshold 5) and sum of lesions
% %----------------------------------------------------------------
% 

% % if (strcmp(site, 'KCH') == 1) 
% %     if strcmp(time_frame, 'retrospective')
% %         value_check = 'True'
% %     elseif strcmp(time_frame, 'prospective')
% %         value_check = 'False'
% %     end
% % end
% cont_size0_lesions = 0
% lesion_files_list = dir(lesionsFolder);
% flag = 0;
% aux = 0;
% count = 0;
% 
% for lsf=1:size(T,1) 
%     aux = aux + 1;
%     if (strcmp(site, 'KCH') == 1) 
% %         if strcmp(T.retrospectiveFlag{lsf}, value_check)
%             lesion_filename = T.lesionPath{lsf};
%             if strcmp(lesion_filename, '#conversion error') == 0
%                 count = count + 1;
%                 [filepath,name,ext] = fileparts(lesion_filename);
%                 resizedlesion = strcat(lesionsFolder, 'rs_', name);
%                 VI = spm_vol(resizedlesion);
%                 img = spm_read_vols(VI);
%                 if(flag == 0)
%                     auxmask = zeros(size(img));
%                     flag = 1;
%                 end
%                 image_indices = find(img~=0);
%                 if (numel(image_indices) == 0)
%                     cont_size0_lesions = cont_size0_lesions + 1;
%                     disp(resizedlesion);
%                 end
%                 auxmask(image_indices) = auxmask(image_indices) + 1;
%             end
%         end
% %     end
% end
% 
% 
nhits = 5
% nhits = round(count*0.002)
% 
sum_img_filename = strcat(matchFolder, site, '__', time_frame, '__sum_lesions_fwhm4_resized4mm.nii')
mask_filename = strcat(matchFolder, site, '__', time_frame, '__mask_lesions_fwhm4_resized4mm_th', int2str(nhits), '.nii');
lesions_table_filename = strcat(matchFolder, site, '__', time_frame, '__LESION_voxels_fwhm4_resized4mm_th', int2str(nhits), '.csv')
% 
% %  
% % Save sum of lesions
% VO = VI;
% VO.fname= sum_img_filename;
% spm_write_vol(VO,auxmask);
% 
% % Save lesion mask
% auxmask(find(auxmask<nhits))=0;
% auxmask(find(auxmask~=0))=1;
% VO = VI;
% VO.fname = mask_filename;
% VO.dt = [8 0];
% spm_write_vol(VO,auxmask);


% for i=3:size(lesion_files_list,1)
%     if (startsWith(lesion_files_list(i).name,'rs_'))   
%             resizedlesion = strcat(lesionsFolder,lesion_files_list(i).name)
%             VI = spm_vol(resizedlesion);
%             img = spm_read_vols(VI);
%             if(flag == 0)
%                 auxmask = zeros(size(img));
%                 flag = 1;
%             end
%             image_indices = find(img~=0);
%             if (numel(image_indices) == 0)
%                 cont_size0_lesions = cont_size0_lesions + 1;
%                 disp(resizedlesion)
%             end
%             auxmask(image_indices) = auxmask(image_indices) + 1;
%     end
% end
% 
% 

%  
 
% ----------------------------------------------------------------
% Step 5: Apply mask and save dataset of lesions as a table
% ----------------------------------------------------------------

% Read mask
VM = spm_vol(mask_filename);
mask = spm_read_vols(VM);
mask_indices = find(mask>0);
nvoxels = numel(mask_indices);

newT=table();
aux = 0;
count = 0;
% sublist = dir(strcat(datasets_path, 'LESIONS'));
nvalid = size(T,1)
newT.imgKey = repmat({''}, nvalid, 1); 
newT.ssnapIndex = repmat({''}, nvalid, 1); 
newT.retrospectiveFlag = repmat({''}, nvalid, 1);
% 
% T.processedLesionPath = repmat({''}, size(T,1), 1);
% 

for lsf=1:size(T,1)
    aux = aux + 1
    if (strcmp(site, 'KCH') == 1) 
%         if strcmp(T.retrospectiveFlag{lsf}, value_check)
            lesion_filename = T.lesionPath{lsf};
            if strcmp(lesion_filename, '#conversion error') == 0
                [filepath,name,ext] = fileparts(lesion_filename);  
            %     lesion_filename_in_mounted_folder = strrep(lesion_filename,'/media/chrisfoulon/DATA1/',datasets_path)
                processsed_lesion_filename = strcat(lesionsFolder, 'rs_', name);
            %     aux = strrep(aux{1}, '/output', '/rs_output');
            %     aux = strrep(aux, '.nii.gz', '.nii');
            %     processed_lesion_name = strcat(datasets_path, 'LESIONS/', aux);
                VI = spm_vol(processsed_lesion_filename);
                img = spm_read_vols(VI);
                masked_img = img(mask_indices);
            %     data_matrix = [data_matrix; masked_img'];
            else
                masked_img = zeros(size(mask_indices));
            end
            newT(aux,4:nvoxels+3) = num2cell(masked_img');
            newT.imgKey{aux} = T.imgKey(lsf);
            newT.ssnapIndex{aux} = T.ssnapIndex(lsf);
            newT.retrospectiveFlag{aux} = T.retrospectiveFlag(lsf);
%         end       
    end
end


writetable(newT, lesions_table_filename);
% % % 
% % % 
% % % % allFiles = T{:,9}
% % % % for i=1:size(allFiles,1)
% % % %     filename = allFiles{i};
% % % %     newFilename = strrep(filename,'input_','');
% % % %     newFilename = strrep(newFilename,'output_','');
% % % %     newFilename = strrep(newFilename,'non_','');
% % % %     newFilename = strrep(newFilename,'linear_','');
% % % %     newFilename = strrep(newFilename,'input_','');
% % % %     newFilename = strrep(newFilename,'co-rigid_','');
% % % %     newFilename = strrep(newFilename,'rigid_','');
% % % %     newFilename = strrep(newFilename,'geomean_','');
% % % %     newFilename = strrep(newFilename,'denoise_','');
% % % %     newFilename = strrep(newFilename,'geomean_','');
% % % %     newFilename = strrep(newFilename,'_bval1000','');
% % % %     newFilename = extractBefore(newFilename,'_v')
% % % % end
% % % % 
% % % % json_file = strcat(data_path,'preproc_dwi_with_attr_and_reports_matching_PatientAge_fixed.json');
% % % % json = jsondecode(fileread(json_file));
% % % % cases = fieldnames(json);
% % % 
% % % 
% % % 
% % % 
% % % 
% % % 
% % % 
