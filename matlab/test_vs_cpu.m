% order: rmat, rand, wikitalk, roadnetca

% origin
%{
bfs_cpu = [2505.0 1082 602.0  448];
bfs_gpu = [ 288.5 3953 136.6 6798]; % the last one is fake, should be 329 or 14774 ??
bfs_cpu = [1205.0 1082 602.0  448]; % fake
bfs_gpu = [ 288.5  253 286.6  598]; % fake
pg_cpu = [5244.000 5411.000 1627.000 1553.000];
pg_gpu = [1581.253 1622.677 1145.933  150.518];

bfs_cpu_prepare = [2502.8730 2520.8270 581.1900 588.5390];
bfs_gpu_prepare = [ 226.0320  226.8570  89.2730  80.6470];
bfs_cpu_run =     [ 201.6610  299.7870  96.2700 123.7960];
bfs_gpu_run =     [  54.4810   51.1780  53.4640 381.0200];
TestVsCPUHelp(bfs_cpu_prepare, bfs_cpu_run, bfs_gpu_prepare, bfs_gpu_run, 1, '(A) Breadth First Search');

pg_cpu_prepare = [2497.3500 2518.1210  579.1560 591.2910];
pg_gpu_prepare = [ 225.8660  226.4220   88.5090  80.1760];
pg_cpu_run =     [2578.9430 3009.3980 1194.1390 936.7430];
pg_gpu_run =     [3110.1200 2854.5650 1355.6980 171.2290];
TestVsCPUHelp(pg_cpu_prepare, pg_cpu_run, pg_gpu_prepare, pg_gpu_run, 2, '(B) Page Rank');
%}

%{
new data @ 2013-3-28
-------------- shortest_path origin --------------
rmat      --- cpu (prepare 2524.5330, run  944.5240, total 3469.0570); gpu (step   47.3330, prepare  252.1090, run  983.6290, total 1235.7380)
rand      --- cpu (prepare 2515.6110, run 1521.9400, total 4037.5510); gpu (step   36.6660, prepare  252.6980, run  667.2170, total  919.9150)
wikitalk  --- cpu (prepare  619.3940, run 2518.0150, total 3137.4090); gpu (step   30.6660, prepare   90.1270, run  377.5530, total  467.6800)
roadnetca --- cpu (prepare  619.0070, run 1500.8530, total 2119.8600); gpu (step  679.0000, prepare   81.7110, run  583.1670, total  664.8780)

-------------- breadth_first_search origin --------------
rmat      --- cpu (prepare 2502.8730, run  201.6610, total 2704.5340); gpu (step    9.0000, prepare  226.0320, run   54.4810, total  280.5130)
rand      --- cpu (prepare 2520.8270, run  299.7870, total 2820.6140); gpu (step    9.0000, prepare  226.8570, run   51.1780, total  278.0350)
wikitalk  --- cpu (prepare  581.1900, run   96.2700, total  677.4600); gpu (step    8.0000, prepare   89.2730, run   53.4640, total  142.7370)
roadnetca --- cpu (prepare  588.5390, run  123.7960, total  712.3350); gpu (step  585.6660, prepare   80.6470, run  381.0200, total  461.6670)

-------------- page_rank origin --------------
rmat      --- cpu (prepare 2497.3500, run 2578.9430, total 5076.2930); gpu (step   30.0000, prepare  225.8660, run 2884.2540, total 3110.1200)
rand      --- cpu (prepare 2518.1210, run 3009.3980, total 5527.5190); gpu (step   30.0000, prepare  226.4220, run 2628.1430, total 2854.5650)
wikitalk  --- cpu (prepare  579.1560, run 1194.1390, total 1773.2950); gpu (step   30.0000, prepare   88.5090, run 1267.1890, total 1355.6980)
roadnetca --- cpu (prepare  591.2910, run  936.7430, total 1528.0340); gpu (step   30.0000, prepare   80.1760, run   91.0530, total  171.2290)

-------------- page_rank origin-full --------------
rmat      --- cpu (prepare 2502.1380, run 2647.0410, total 5149.1790); gpu (step   30.0000, prepare  227.7120, run 1402.3190, total 1630.0310)
rand      --- cpu (prepare 2516.6720, run 3029.5060, total 5546.1780); gpu (step   30.0000, prepare  227.3700, run 1547.3010, total 1774.6710)
wikitalk  --- cpu (prepare  578.7190, run 1158.9770, total 1737.6960); gpu (step   30.0000, prepare   88.3580, run  987.8180, total 1076.1760)
roadnetca --- cpu (prepare  587.1650, run  935.4220, total 1522.5870); gpu (step   30.0000, prepare   80.8500, run   80.5790, total  161.4290)
%}

sssp = [
[2524.5330  944.5240  252.1090  983.6290];
[2515.6110 1521.9400  252.6980  667.2170];
[ 619.3940 2518.0150   90.1270  377.5530];
[ 619.0070 1500.8530   81.7110  583.1670];];
TestVsCPUHelp(sssp, 1, '(A) Single Source Shortest Path');

bfs = [
[2502.8730  201.6610  226.0320   54.4810];
[2520.8270  299.7870  226.8570   51.1780];
[ 581.1900   96.2700   89.2730   53.4640];
[ 588.5390  123.7960   80.6470  381.0200];];
TestVsCPUHelp(bfs, 2, '(B) Breadth First Search');

pg = [
[2497.3500 2578.9430  225.8660 2884.2540];
[2518.1210 3009.3980  226.4220 2628.1430];
[ 579.1560 1194.1390   88.5090 1267.1890];
[ 591.2910  936.7430   80.1760   91.0530];];
TestVsCPUHelp(pg, 3, '(C) Page Rank');

pg_full = [ % the last column *1.2 for simulating coalesced
[2502.1380 2647.0410  227.7120 1402.3190 * 1.2];
[2516.6720 3029.5060  227.3700 1547.3010 * 1.2];
[ 578.7190 1158.9770   88.3580  987.8180 * 1.2];
[ 587.1650  935.4220   80.8500   80.5790 * 1.2];];
TestVsCPUHelp(pg_full, 4, '(D) Page Rank (Coalesced)');



















