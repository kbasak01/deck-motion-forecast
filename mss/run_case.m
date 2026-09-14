% run_case.m -- drive MSS waveMotionRAO over one realization and dump 6-DOF motion.
%
% Phase 8 parity check. Reads a case file written by Python (so both backends see exactly
% the same wave grid and the same random phases), calls MSS's own waveMotionRAO over the
% time axis, and writes the result as CSV for comparison against src/dmf/mss/synth.py.
%
% Usage:  octave --no-gui --quiet --eval "run_case('case.mat','out.csv')"
%
% Units out: eta rows 1-3 metres, rows 4-6 RADIANS; eta_dot likewise per second. MSS sign
% convention (SNAME, z down) -- conversion to corpus units/signs happens in Python.

function run_case(case_path, out_path)

  here = fileparts(mfilename('fullpath'));
  addpath(genpath(fullfile(here, 'upstream', 'LIBRARY')));
  addpath(here);

  S = load(case_path);
  % Load the vessel structure from the MSS file directly. Round-tripping it through
  % scipy.io.savemat turns the RAO cell arrays into something Octave indexes differently,
  % so Python passes only the wave grid and Octave reads the hull data itself.
  V = load(fullfile(here, 'upstream', 'HYDRO', 'vessels_shipx', 's175', 's175.mat'));
  vessel = V.vessel;
  Omega  = S.Omega(:);
  Amp    = S.Amp(:);
  phases = S.phases(:);
  t      = S.t(:);
  U      = double(S.U);
  beta   = double(S.beta_wave);
  nFreq  = numel(Omega);

  % The RAO frequency interpolation inside waveMotionRAO is cached in a `persistent`
  % variable and is only computed when it is empty, so calling it with a DIFFERENT Omega
  % grid in the same session would silently reuse the previous interpolation. Clear it.
  clear -f waveMotionRAO_seeded;

  S_M = ones(nFreq, 1);   % only size(S_M,2)==1 is used: selects the no-spreading branch
  mu  = 0;

  n = numel(t);
  eta     = zeros(6, n);
  eta_dot = zeros(6, n);
  elev    = zeros(1, n);

  for k = 1:n
    [e, v, ~, w] = waveMotionRAO_seeded( ...
        t(k), S_M, Amp, Omega, mu, vessel, U, 0.0, beta, nFreq, phases);
    eta(:, k)     = e;
    eta_dot(:, k) = v;
    elev(k)       = w;
  end

  out = [t(:), eta.', eta_dot.', elev(:)];
  hdr = {'t','eta1','eta2','eta3','eta4','eta5','eta6', ...
         'etadot1','etadot2','etadot3','etadot4','etadot5','etadot6','elevation'};
  fid = fopen(out_path, 'w');
  fprintf(fid, '%s', hdr{1});
  fprintf(fid, ',%s', hdr{2:end});
  fprintf(fid, '\n');
  fclose(fid);
  dlmwrite(out_path, out, '-append', 'precision', '%.12g');

  printf('wrote %d rows to %s\n', n, out_path);
end
