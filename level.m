function [ level ] = level(nLevel, ringSize)
%Returns level number if Baseline, returns 0
% level 1 - big ring & 1-back 
% level 2 - medium ring & 1-back
% level 3 - medium ring & 2-back
% level 4 - small ring & 2-back
% level 5 - small ring & 3-back
% 
% Baselines: 
% big ring (no nback)
% medium ring  (no nback)
% small ring (no nback)
% 0-back (no rings)
% if no nback : nLevel == -1
% if no rings : ringSize == 'no'

level = 0;

if (strcmp(ringSize, 'big') && strcmp(nLevel, '1'))
    level = 1;
elseif (strcmp(ringSize, 'medium') && strcmp(nLevel, '1'))
    level = 2;
elseif (strcmp(ringSize, 'medium') && strcmp(nLevel, '2'))
    level = 3;
elseif (strcmp(ringSize, 'small') && strcmp(nLevel, '2'))
    level = 4;
elseif (strcmp(ringSize, 'small') && strcmp(nLevel, '3'))
    level = 5;
elseif (strcmp(ringSize, 'big') && strcmp(nLevel, '-1'))
    level = 0; %Baseline
elseif (strcmp(ringSize, 'medium') && strcmp(nLevel, '-1'))
    level = 0; %Baseline
elseif (strcmp(ringSize, 'small') && strcmp(nLevel, '-1'))
    level = 0; %Baseline
elseif (strcmp(ringSize, 'no') && strcmp(nLevel, '0'))
    level = 0; %Baseline
else
    error('Problem with Level');
end


