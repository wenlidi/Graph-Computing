% old data, 50000v 100000-150000e 5000s
%{
origin_rmat =    [85.274 94.629 109.138 122.030 138.656 152.271 164.179 182.542 189.454 211.206 227.092];
coalesced_rmat = [80.748 91.562 104.282 115.881 130.000 141.037 157.286 167.000 177.000 189.103 203.922];
% fake                                                                    0.537   0.466

origin_rand =    [71.195 77.648 84.383 91.070 97.275 105.561 111.728 119.461 125.723 134.983 149.210];
coalesced_rand = [64.354 68.882 72.457 76.189 80.492  83.093  86.754  90.000  93.000  97.481 100.672];
% fake                                                                 0.473   0.466
%}

% new data @ 2013-3-28, 10^6v 10^6-16*10^6e 30s
%{
rmat:
--------------- origin rolling ---------------
Super step prepare 0:    41.354 ms,    Super step run 0:   100.642 ms
Super step prepare 0:    54.386 ms,    Super step run 0:   221.808 ms
Super step prepare 0:    65.101 ms,    Super step run 0:   357.723 ms
Super step prepare 0:    78.547 ms,    Super step run 0:   507.313 ms
Super step prepare 0:    90.463 ms,    Super step run 0:   669.338 ms
Super step prepare 0:   103.014 ms,    Super step run 0:   843.542 ms
Super step prepare 0:   115.819 ms,    Super step run 0:  1023.526 ms
Super step prepare 0:   125.762 ms,    Super step run 0:  1214.099 ms
Super step prepare 0:   141.074 ms,    Super step run 0:  1412.295 ms
Super step prepare 0:   151.321 ms,    Super step run 0:  1610.860 ms
Super step prepare 0:   165.269 ms,    Super step run 0:  1812.922 ms
Super step prepare 0:   177.447 ms,    Super step run 0:  2024.642 ms
Super step prepare 0:   189.290 ms,    Super step run 0:  2230.968 ms
Super step prepare 0:   203.736 ms,    Super step run 0:  2449.406 ms
Super step prepare 0:   215.359 ms,    Super step run 0:  2669.556 ms
Super step prepare 0:   227.848 ms,    Super step run 0:  2882.597 ms

--------------- coalesced rolling full ---------------
Super step prepare 0:    60.940 ms,    Super step run 0:    65.025 ms
Super step prepare 0:    83.803 ms,    Super step run 0:   140.936 ms
Super step prepare 0:   105.129 ms,    Super step run 0:   224.943 ms
Super step prepare 0:   127.637 ms,    Super step run 0:   315.143 ms
Super step prepare 0:   149.110 ms,    Super step run 0:   408.707 ms
Super step prepare 0:   171.976 ms,    Super step run 0:   505.868 ms
Super step prepare 0:   197.609 ms,    Super step run 0:   604.571 ms
Super step prepare 0:   223.360 ms,    Super step run 0:   704.803 ms
Super step prepare 0:   248.751 ms,    Super step run 0:   807.153 ms
Super step prepare 0:   274.221 ms,    Super step run 0:   909.450 ms
Super step prepare 0:   299.542 ms,    Super step run 0:  1013.803 ms
Super step prepare 0:   319.846 ms,    Super step run 0:  1116.044 ms
Super step prepare 0:   356.416 ms,    Super step run 0:  1219.348 ms
Super step prepare 0:   371.804 ms,    Super step run 0:  1324.703 ms
Super step prepare 0:   403.360 ms,    Super step run 0:  1429.566 ms
Super step prepare 0:   427.073 ms,    Super step run 0:  1535.778 ms

rand:
--------------- origin rolling ---------------
Super step prepare 0:    40.756 ms,    Super step run 0:   106.332 ms
Super step prepare 0:    52.263 ms,    Super step run 0:   222.931 ms
Super step prepare 0:    66.425 ms,    Super step run 0:   344.969 ms
Super step prepare 0:    79.456 ms,    Super step run 0:   474.311 ms
Super step prepare 0:    89.028 ms,    Super step run 0:   610.544 ms
Super step prepare 0:   102.583 ms,    Super step run 0:   755.812 ms
Super step prepare 0:   113.427 ms,    Super step run 0:   911.505 ms
Super step prepare 0:   128.085 ms,    Super step run 0:  1079.687 ms
Super step prepare 0:   142.815 ms,    Super step run 0:  1253.370 ms
Super step prepare 0:   153.222 ms,    Super step run 0:  1431.110 ms
Super step prepare 0:   164.598 ms,    Super step run 0:  1616.045 ms
Super step prepare 0:   177.545 ms,    Super step run 0:  1807.350 ms
Super step prepare 0:   188.854 ms,    Super step run 0:  2002.914 ms
Super step prepare 0:   203.941 ms,    Super step run 0:  2206.536 ms
Super step prepare 0:   213.499 ms,    Super step run 0:  2414.993 ms
Super step prepare 0:   227.204 ms,    Super step run 0:  2629.289 ms

--------------- coalesced rolling full ---------------
Super step prepare 0:    62.311 ms,    Super step run 0:    59.809 ms
Super step prepare 0:    84.463 ms,    Super step run 0:   119.279 ms
Super step prepare 0:   106.678 ms,    Super step run 0:   182.583 ms
Super step prepare 0:   127.815 ms,    Super step run 0:   250.062 ms
Super step prepare 0:   147.240 ms,    Super step run 0:   319.734 ms
Super step prepare 0:   166.807 ms,    Super step run 0:   393.540 ms
Super step prepare 0:   188.186 ms,    Super step run 0:   470.083 ms
Super step prepare 0:   208.111 ms,    Super step run 0:   549.205 ms
Super step prepare 0:   228.822 ms,    Super step run 0:   631.234 ms
Super step prepare 0:   248.952 ms,    Super step run 0:   716.976 ms
Super step prepare 0:   269.484 ms,    Super step run 0:   804.625 ms
Super step prepare 0:   289.996 ms,    Super step run 0:   893.501 ms
Super step prepare 0:   310.378 ms,    Super step run 0:   987.076 ms
Super step prepare 0:   332.040 ms,    Super step run 0:  1081.863 ms
Super step prepare 0:   350.979 ms,    Super step run 0:  1180.372 ms
Super step prepare 0:   375.104 ms,    Super step run 0:  1280.618 ms
%}

origin_rmat_prepare = [ 41.354 54.386 65.101 78.547 90.463 103.014 115.819 125.762 141.074 151.321 165.269 177.447 189.290 203.736 215.359 227.848];
origin_rmat_run = [ 100.642 221.808 357.723 507.313 669.338 843.542 1023.526 1214.099 1412.295 1610.860 1812.922 2024.642 2230.968 2449.406 2669.556 2882.597];

coalesced_rmat_prepare = [ 60.940 83.803 105.129 127.637 149.110 171.976 197.609 223.360 248.751 274.221 299.542 319.846 356.416 371.804 403.360 427.073];
coalesced_rmat_run = [ 65.025 140.936 224.943 315.143 408.707 505.868 604.571 704.803 807.153 909.450 1013.803 1116.044 1219.348 1324.703 1429.566 1535.778];

origin_rand_prepare = [ 40.756 52.263 66.425 79.456 89.028 102.583 113.427 128.085 142.815 153.222 164.598 177.545 188.854 203.941 213.499 227.204];
origin_rand_run = [  106.332 222.931 344.969 474.311 610.544 755.812 911.505 1079.687 1253.370 1431.110 1616.045 1807.350 2002.914 2206.536 2414.993 2629.289];

coalesced_rand_prepare = [  62.311 84.463 106.678 127.815 147.240 166.807 188.186 208.111 228.822 248.952 269.484 289.996 310.378 332.040 350.979 375.104];
coalesced_rand_run = [  59.809 119.279 182.583 250.062 319.734 393.540 470.083 549.205 631.234 716.976 804.625 893.501 987.076 1081.863 1180.372 1280.618];

x = 1000000:1000000:16000000;

% 出边不均匀，所以看起来没那么大差别，因为受出边的影响较大
subplot(1, 2, 1);
Test2AlgorithmPrepareRun(...
    origin_rmat_prepare,...
    origin_rmat_run,...
    coalesced_rmat_prepare,...
    coalesced_rmat_run,...
    [1000000 16000000], x,...
    '(A) R-MAT', 'Number of Edges', 'Execution Time (ms)', 'Basic', 'Coalesced')
%{
plot(x, origin_rmat_prepare, 'ro:', ...
     x, coalesced_rmat_prepare, 'bs:', ...
     x, origin_rmat_run, 'r+-', ...
     x, coalesced_rmat_run, 'b*-');
set(gca, 'XLim', [1000000 16000000]);
set(gca, 'XTick', x);
title('(A) R-MAT');
xlabel('Number of Edges');
ylabel('Execution Time (ms)');
legend('Basic (Preparing)', ...
       'Coalesced (Preparing)', ...
       'Basic (Compute)', ...
       'Coalesced (Compute)', ...
       'Location', 'NorthWest')
%}

subplot(1, 2, 2);
Test2AlgorithmPrepareRun(...
    origin_rand_prepare,...
    origin_rand_run,...
    coalesced_rand_prepare,...
    coalesced_rand_run,...
    [1000000 16000000], x,...
    '(B) Random', 'Number of Edges', 'Execution Time (ms)', 'Basic', 'Coalesced')
%{
plot(x, origin_rand_prepare, 'ro:', ...
     x, coalesced_rand_prepare, 'bs:', ...
     x, origin_rand_run, 'r+-', ...
     x, coalesced_rand_run, 'b*-');
set(gca, 'XLim', [1000000 16000000]);
set(gca, 'XTick', x);
title('(B) Random');
xlabel('Number of Edges');
ylabel('Execution Time (ms)');
legend('Basic (Preparing)', ...
       'Coalesced (Preparing)', ...
       'Basic (Compute)', ...
       'Coalesced (Compute)', ...
       'Location', 'NorthWest')
%}