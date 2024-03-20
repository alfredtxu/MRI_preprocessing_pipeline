% List of open inputs
nrun = X; % enter the number of runs here
jobfile = {'/home/jrondina/Desktop/PycharmProjects/predictive-tool/smooth_lesions_job.m'};
jobs = repmat(jobfile, 1, nrun);
inputs = cell(0, nrun);
for crun = 1:nrun
end
spm('defaults', 'PET');
spm_jobman('run', jobs, inputs{:});
