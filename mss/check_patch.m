% check_patch.m -- verify waveMotionRAO_seeded is behaviour-preserving.
%
% Runs the STOCK upstream waveMotionRAO.m (which seeds rng(12345,"twister") internally) and
% the patched copy fed with exactly the phases stock would have drawn, then reports the
% largest disagreement. If the patch changed anything beyond the source of the phases, this
% is where it shows up. Writes a one-line CSV so Python can read the verdict.

function check_patch(case_path, out_path)

  here = fileparts(mfilename('fullpath'));
  addpath(genpath(fullfile(here, 'upstream', 'LIBRARY')));
  addpath(here);

  S = load(case_path);
  V = load(fullfile(here, 'upstream', 'HYDRO', 'vessels_shipx', 's175', 's175.mat'));
  vessel = V.vessel;
  Omega = S.Omega(:); Amp = S.Amp(:); t = S.t(:);
  U = double(S.U); beta = double(S.beta_wave); nFreq = numel(Omega);
  S_M = ones(nFreq, 1); mu = 0;

  % Reproduce the exact draw stock makes on its first call.
  rng(12345, "twister");
  stockPhases = 2 * pi * rand(nFreq, 1);

  clear -f waveMotionRAO;
  clear -f waveMotionRAO_seeded;

  n = numel(t);
  worst = 0.0; scale = 0.0;
  for k = 1:n
    [e1, v1, ~, w1] = waveMotionRAO( ...
        t(k), S_M, Amp, Omega, mu, vessel, U, 0.0, beta, nFreq);
    [e2, v2, ~, w2] = waveMotionRAO_seeded( ...
        t(k), S_M, Amp, Omega, mu, vessel, U, 0.0, beta, nFreq, stockPhases);
    d = max([abs(e1 - e2); abs(v1 - v2); abs(w1 - w2)]);
    s = max([abs(e1); abs(v1); abs(w1)]);
    worst = max(worst, d); scale = max(scale, s);
  end

  rel = worst / max(scale, 1e-30);
  fid = fopen(out_path, 'w');
  fprintf(fid, 'max_abs_diff,scale,rel_diff\n%.17g,%.17g,%.17g\n', worst, scale, rel);
  fclose(fid);
  printf('patch equivalence: max_abs_diff=%.3e rel=%.3e over %d steps\n', worst, rel, n);
end
